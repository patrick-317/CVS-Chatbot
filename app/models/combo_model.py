import os
import re
from typing import List, Dict, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------- 경로 설정 --------------------
# combo_model.py 위치: CVS-Chatbot/app/models/combo_model.py
# 프로젝트 루트: CVS-Chatbot
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")

COMB_PATH = os.path.join(DATA_DIR, "combination.csv")
SYN_PATH = os.path.join(DATA_DIR, "synthetic_honey_combos_1000.csv")
SAVE_PATH = os.path.join(PRECOMPUTED_DIR, "product2vec.pt")

# -------------------- 하이퍼파라미터 --------------------

EMBED_DIM = 64
WINDOW_SIZE = 3
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- 데이터셋에서 상품 시퀀스 추출 --------------------


def combos_from_df(df: pd.DataFrame) -> List[List[str]]:
    """
    combination.csv / synthetic_honey_combos_1000.csv 공통 스키마:

    컬럼:
      - '조합 이름'
      - '편의점'
      - '주요 상품'
      - '보조 상품(들)'
      - '키워드 / 상황'
      - '카테고리'
      - '원문 URL (출처)'

    여기서:
      - '주요 상품'  → 반드시 1개 상품
      - '보조 상품(들)' → "A, B, C" 처럼 콤마로 구분된 상품들

    → 각 row 를 ["주요 상품", "보조1", "보조2", ...] 형태의 시퀀스로 변환
    """
    main_col = "주요 상품"
    sub_col = "보조 상품(들)"

    if main_col not in df.columns or sub_col not in df.columns:
        print("[product2vec] 예상한 컬럼이 없습니다. columns =", df.columns.tolist())
        return []

    sentences: List[List[str]] = []

    for _, row in df.iterrows():
        items: List[str] = []

        # 1) 주요 상품
        main = str(row.get(main_col, "")).strip()
        if main and main.lower() != "nan":
            items.append(main)

        # 2) 보조 상품(들) - 콤마 기준으로 나눔
        subs_raw = str(row.get(sub_col, "")).strip()
        if subs_raw and subs_raw.lower() != "nan":
            parts = [p.strip() for p in subs_raw.split(",") if p.strip()]
            items.extend(parts)

        # 3) 최소 2개 이상이어야 "조합"으로 의미가 있음
        if len(items) < 2:
            continue

        # 4) 중복 제거(순서 유지)
        uniq: List[str] = []
        seen = set()
        for it in items:
            if it not in seen:
                seen.add(it)
                uniq.append(it)

        if len(uniq) >= 2:
            sentences.append(uniq)

    return sentences


def load_sentences() -> List[List[str]]:
    """
    실제/AI 조합 둘 다 로드해서 하나의 코퍼스로 사용
    """
    print("[product2vec] COMB_PATH =", COMB_PATH)
    print("[product2vec] SYN_PATH  =", SYN_PATH)

    df_comb = pd.read_csv(COMB_PATH)
    df_syn = pd.read_csv(SYN_PATH)

    s_comb = combos_from_df(df_comb)
    s_syn = combos_from_df(df_syn)

    sentences = s_comb + s_syn
    print(f"[product2vec] sentences_comb = {len(s_comb)}")
    print(f"[product2vec] sentences_syn  = {len(s_syn)}")
    print(f"[product2vec] sentences_total = {len(sentences)}")

    return sentences


# -------------------- Dataset / Model --------------------


class Product2VecDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: Dict[str, int]):
        self.data: List[Tuple[int, int]] = []
        self.vocab = vocab

        # (center, context) 쌍 생성
        for sent in sentences:
            idxs = [vocab[w] for w in sent if w in vocab]
            for i, center in enumerate(idxs):
                left = max(0, i - WINDOW_SIZE)
                right = min(len(idxs), i + WINDOW_SIZE + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    context = idxs[j]
                    self.data.append((center, context))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        c, t = self.data[idx]
        return torch.tensor(c, dtype=torch.long), torch.tensor(t, dtype=torch.long)


class Product2Vec(nn.Module):
    """
    아주 단순한 product2vec (word2vec 유사 구조)
    - in_emb: center
    - out_emb: context
    - 손실: (dot(center, context) - 1)^2
    """

    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.out_emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, center_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        v_c = self.in_emb(center_idx)   # (B, D)
        v_t = self.out_emb(target_idx)  # (B, D)
        scores = (v_c * v_t).sum(dim=1)  # (B,)
        return scores


# -------------------- 학습 루프 --------------------


def build_vocab(sentences: List[List[str]]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for sent in sentences:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)
    print(f"[product2vec] vocab_size = {len(vocab)}")
    return vocab


def train():
    sentences = load_sentences()
    if not sentences:
        print("[product2vec] 유효한 조합 문장이 없습니다. CSV 경로나 컬럼을 다시 확인해주세요.")
        return

    vocab = build_vocab(sentences)
    if not vocab:
        print("[product2vec] vocab이 비어 있습니다. 데이터 전처리를 다시 확인해주세요.")
        return

    dataset = Product2VecDataset(sentences, vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = Product2Vec(vocab_size=len(vocab), emb_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for center_idx, target_idx in dataloader:
            center_idx = center_idx.to(DEVICE)
            target_idx = target_idx.to(DEVICE)

            optimizer.zero_grad()
            logits = model(center_idx, target_idx)  # (B,)

            # target dot-product 를 1에 가깝게
            loss = ((logits - 1.0) ** 2).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[product2vec] epoch {epoch + 1}/{EPOCHS}, loss = {total_loss:.4f}")

    # CPU로 임베딩 추출
    with torch.no_grad():
        emb = model.in_emb.weight.detach().cpu().clone()

    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    torch.save(
        {
            "embeddings": emb,
            "vocab": vocab,
            "emb_dim": EMBED_DIM,
        },
        SAVE_PATH,
    )
    print("[product2vec] saved to", SAVE_PATH)


if __name__ == "__main__":
    train()
