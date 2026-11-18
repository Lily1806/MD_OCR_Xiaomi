import os, sys, json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

SEARCH_ROOT = Path(os.environ.get('CC_SEARCH_ROOT', '/kaggle/input'))
IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def prepare_env():
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def patch_peft_torchao():
    for _n, _m in list(sys.modules.items()):
        if _n.startswith('peft') and hasattr(_m, 'is_torchao_available'):
            _m.is_torchao_available = lambda: False


def find_file(name, root):
    root = Path(root)
    if (root / name).exists():
        return root / name
    hits = list(root.rglob(name)) if root.exists() else []
    if hits:
        return hits[0]
    hits = list(SEARCH_ROOT.rglob(name)) if SEARCH_ROOT.exists() else []
    if hits:
        return hits[0]
    raise FileNotFoundError(f'Khong tim thay {name} trong {root}')


def _content_text(content):
    if isinstance(content, list):
        return ''.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content


def get_label(rec):
    lab = rec.get('label')
    if lab is not None:
        return lab if isinstance(lab, str) else json.dumps(lab, ensure_ascii=False)
    return _content_text(rec['messages'][-1]['content'])


def assistant_text(rec):
    return _content_text(rec['messages'][-1]['content'])


def clean_messages(msgs):
    out = []
    for turn in msgs:
        if turn['role'] == 'user' and isinstance(turn['content'], list):
            content = [{'type': 'image'} if p.get('type') == 'image' else p for p in turn['content']]
            out.append({'role': turn['role'], 'content': content})
        else:
            out.append(turn)
    return out


class OCRDataset(Dataset):
    def __init__(self, jsonl_path, root, prompt_override=None):
        self.root = Path(root)
        p = find_file(Path(jsonl_path).name, self.root)
        self.jsonl_path = p
        self.jsonl_dir = p.parent
        with open(p, encoding='utf-8') as f:
            self.records = [json.loads(l) for l in f if l.strip()]
        if prompt_override:
            for rec in self.records:
                for turn in rec['messages']:
                    if turn['role'] == 'user' and isinstance(turn['content'], list):
                        for part in turn['content']:
                            if part.get('type') == 'text':
                                part['text'] = prompt_override
        self._index = None
        first = self._resolve(self.records[0]['image'])
        print(f'[Dataset] {len(self.records)} records <- {p}  (anh mau: {first})', flush=True)

    def _resolve(self, rel):
        for base in (self.jsonl_dir, self.root, self.jsonl_dir.parent):
            c = base / rel
            if c.exists():
                return c
        if self._index is None:
            self._index = {}
            for base in {self.root, self.jsonl_dir}:
                for f in Path(base).rglob('*'):
                    if f.is_file() and f.suffix.lower() in IMG_EXT:
                        self._index.setdefault(f.name, f)
        name = Path(rel).name
        if name in self._index:
            return self._index[name]
        raise FileNotFoundError(f'Khong thay anh: {rel}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(self._resolve(rec['image'])).convert('RGB')
        return {'image': img, 'label': get_label(rec), 'messages': rec['messages']}

    def resolve(self, rel):
        return self._resolve(rel)


class OCRDataCollator:
    """Chi tinh loss tren phan tra loi cua assistant.
    Uu tien: sau the </think> cuoi cung. Du phong: sau '<|im_start|>assistant'."""

    def __init__(self, processor, max_length=1024):
        self.processor = processor
        self.max_length = max_length
        self.tok = processor.tokenizer
        unk = self.tok.unk_token_id
        def _id(t):
            i = self.tok.convert_tokens_to_ids(t)
            return None if (i is None or (unk is not None and i == unk)) else i
        self.im_end_id = _id('<|im_end|>')
        self.im_start_id = _id('<|im_start|>')
        self.think_end_id = _id('</think>')
        self._warned = False
        self.n_empty = 0

    def _find_span(self, l):
        ends = [j for j, t in enumerate(l) if t == self.im_end_id] if self.im_end_id is not None else []
        if not ends:
            return None
        start = None
        if self.think_end_id is not None:
            think = [j for j, t in enumerate(l) if t == self.think_end_id]
            if think:
                start = think[-1] + 1
        if start is None and self.im_start_id is not None:
            st = [j for j, t in enumerate(l) if t == self.im_start_id]
            if st:
                start = st[-1] + 1
                # bo qua 'assistant' va xuong dong
                while start < len(l) and l[start] != self.im_end_id and \
                        self.tok.decode([l[start]]).strip() in ('', 'assistant'):
                    start += 1
        if start is None:
            return None
        while start < len(l) and l[start] != self.im_end_id and self.tok.decode([l[start]]).strip() == '':
            start += 1
        end = ends[-1]
        return (start, end) if start <= end else None

    def __call__(self, batch):
        images = [b['image'] for b in batch]
        texts = [self.processor.apply_chat_template(clean_messages(b['messages']),
                                                    tokenize=False, add_generation_prompt=False)
                 for b in batch]
        enc = self.processor(text=texts, images=images, return_tensors='pt', padding=True)
        if enc['input_ids'].shape[1] > self.max_length and not self._warned:
            print(f"Chuoi dai {enc['input_ids'].shape[1]} > {self.max_length} token", flush=True)
            self._warned = True
        ids_all = enc['input_ids']
        labels = torch.full_like(ids_all, -100)
        for i, ids in enumerate(ids_all):
            span = self._find_span(ids.tolist())
            if span is None:
                self.n_empty += 1
                continue
            s, e = span
            labels[i, s:e + 1] = ids[s:e + 1]
        enc['labels'] = labels
        return enc