# Lab 21 - Evaluation Report

**Ho ten**: Ngo Hung Phuc  **MSSV**: 2A202601069  **Ngay**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thuc te**: `Colab T4`

> Moi con so duoi day duoc lay tu cac file trong `results/`. Lan chay nay dung
> `EVAL_LIMIT=8` nen day la smoke/evaluation rut gon,
> khong phai full eval mac dinh.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tieng Viet -> JSON triage 4 truong |
| Train / val | 225 / 25, split seed 42 |
| `max_length` | 1024 - p95 do duoc la 98, suggested max_length = 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | default 2 epoch / 30 optimizer steps |

**Template co giu khoi `<think>` khong?** Co. `template_check.json` co `ok=true`,
`open_tag_present=true`, `body_present=true`, va verdict la `reasoning preserved — safe to train on traces`.
Nghia la chat template khong nuot mat reasoning trace, nen neu dataset co trace thi
phan trace van co the di qua tokenizer va den duoc loss mask. Trong bai nay em giu
`MASK_MODE=assistant-only`, khong doi template hay prompt danh gia.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Cau tra loi nam trong loss | `true` |
| Cau hoi KHONG nam trong loss | `true` |

3-5 dong dau cua doan duoc tinh loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Ket qua nay quan trong hon duong loss ve sau: chi 41.49% token cua sample dau tien
duoc supervise, va doan supervised bat dau o luot assistant, chua JSON answer. Phan
user/question nam trong `masked_preview`, khong nam trong loss. Neu dung `everything`,
model se hoc viet lai cau hoi, nen em giu `assistant-only` cho toan bo run.

---

## 3. Ba baseline (NB2 - do TRUOC khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3267.6 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 1000.2 |
| (c) LoRA fine-tune | 0.9375 | 0.8750 | 1.0000 | 1778.4 |

**(b) co that su manh hon (a) khong?** Co. Base + naive prompt dat target 0.0000
va format 0.0000, trong khi base + optimized prompt dat target 0.6875 va format
1.0000, regression giu nguyen 0.7500. Dieu nay chung minh baseline (b) la baseline
nghiem tuc, khong bi lam yeu de fine-tune trong co ve thang. Em khong sua
`OPTIMIZED_PROMPT`; `optimized_prompt_sha` trong `baselines_frozen.json` la
`719e74d3b6232053`.

---

## 4. Giai phau cau hinh sai (NB4)

| Run | vi tri | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6267 | 0.9375 | 926.3 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5371 | 0.9375 | 775.7 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | 0.0000 | 1033.0 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.8438 | 1125.8 | 7.09 |

**4.1 - `attn_only` co cung so tham so huan luyen voi `correct`.** Tren tap target
rut gon, `attn_only` hoa voi `correct`: ca hai deu dat target 0.9375. Tuy nhien thu
tu theo train loss lai khac: `attn_only` co final_loss 0.5371, thap hon `correct`
0.6267, nhung target khong cao hon. Dieu nay noi rang final loss khong du de ket
luan adapter tot hon tren task thuc. Khi rank cua `attn_only` duoc nang len 283 de
match ngan sach tham so voi `correct`, viec no khong vuot `correct` cho thay rank
lon hon khong tu dong la don bay chinh; vi tri gan adapter va metric target moi la
bang chung can doc.

**4.2 - `wrong_lr` chi khac dung mot con so.** `wrong_lr` dung LR 1e-05 thay vi
0.0001, final_loss tang len 1.5704 va target roi ve 0.0000. Neu chi nhin mot cach
nong can, co the ket luan sai rang LoRA khong hoc duoc hoac dataset qua kho. Thuc
te doi chung nay cho thay learning rate scale la bien rat manh: cung placement
text-linear, cung r=16, cung so step 30, nhung LR o thang full fine-tune lam
training gan nhu khong dat task. Duong loss cao hon va target fail la bang chung
rang cau hinh LoRA can LR lon hon full fine-tune.

**4.3 - `qlora` tiet kiem VRAM nhung tra gia bang chat luong.** `qlora` dung 7.09
GB peak VRAM so voi `correct` 12.01 GB, tiet kiem khoang 4.92 GB, tuc gan 41.0%
VRAM. Doi lai target giam tu 0.9375 xuong 0.8438 va final_loss cao hon `correct`
(0.7058 so voi 0.6267). Trong smoke run nay QLoRA van chay duoc va format van
1.0000, nhung no khong thang cau hinh 16-bit LoRA. So do ung ho khuyen nghi cua
lab: voi dong Qwen3.5 nay, QLoRA nen duoc do nhu mot contrast run, khong nen chon
lam mac dinh neu T4 van du VRAM cho LoRA 16-bit.

---

## 5. Phan quyet (NB5)

**Ket qua cong hoi quy**: `PASSED`
`target Δ = +0.250` ·
`regression Δ = +0.125` ·
`valid_trace_rate = 0.00`

Fine-tune PASS trong lan chay rut gon nay. Target cua LoRA fine-tune dat 0.9375,
cao hon optimized prompt baseline 0.6875, tuc tang +0.250. Regression cung tang tu
0.7500 len 0.8750, nen khong co dau hieu ro rang ve viec fine-tune lam hong nang
luc chung tren slice regression nay. Format giu 1.0000, nghia la adapter hoc duoc
dang JSON 4 khoa tot hon base naive va khong lam vo constraint dau ra. Tuy vay, can
doc ket qua nay voi dieu kien: `baselines_frozen.json` ghi `smoke_mode=true` va
`eval_limit=8`, nen day la bang chung nhanh tren 8 mau, chua phai ket luan san
sang deploy. Neu nop theo rubric day du, can rerun khong set `EVAL_LIMIT` de dung
full eval set. Dieu co gia tri nhat o day la phuong phap phan quyet: so voi
baseline (b) truoc train, doc them regression, format, latency, khong chi nhin
final loss.

---

## 6. Dinh tinh - bat buoc co ca ca THUA

| # | Ticket (rut gon) | Nhan dung | (b) prompt | (c) fine-tune | Nhan xet |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền.... | intent=hoan_tien, urgency=thap, product=bình giữ nhiệt, sentiment=tich_cuc | khong luu prediction chi tiet trong qualitative.json | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "s...` | FT thua mot phan, ft_score=0.75 |
| 2 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện.... | intent=san_pham_loi, urgency=thap, product=nồi chiên không dầu, sentiment=trung_tinh | khong luu prediction chi tiet trong qualitative.json | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không ...` | FT thua mot phan, ft_score=0.75 |
| 3 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lạ... | intent=doi_tra, urgency=cao, product=chuột không dây, sentiment=tich_cuc | khong luu prediction chi tiet trong qualitative.json | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment...` | FT thang/gan dung, ft_score=1.00 |
| 4 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm n... | intent=hoan_tien, urgency=trung_binh, product=ốp lưng điện thoại, sentiment=tieu_cuc | khong luu prediction chi tiet trong qualitative.json | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại"...` | FT thang/gan dung, ft_score=1.00 |
| 5 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | intent=hoan_tien, urgency=cao, product=đèn bàn LED, sentiment=tich_cuc | khong luu prediction chi tiet trong qualitative.json | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment":...` | FT thang/gan dung, ft_score=1.00 |

Cac ca FT thua mot phan tap trung o nhung mau ma output bi cat/truncated trong
`ft_pred`. Vi du i=3 va i=5 chi dat 0.75: model bat dau dung intent/product nhung
chuoi JSON chua ket thuc day du trong preview, nen co kha nang mat diem o mot truong
hoac bi anh huong boi generation length/format. Mau chung la fine-tune da hoc schema
tot hon baseline naive, nhung van can audit decode settings va max_new_tokens de
tranh cat JSON giua chung.

---

## 7. Ket luan & dieu toi hoc duoc

**Ket luan.** Trong lan chay smoke voi T4, em se chua deploy ngay ban fine-tune nay
cho khach hang that, nhung em xem no la mot ung vien tot de tiep tuc danh gia full
set. Ly do la LoRA correct thang baseline prompt toi uu tren target (+0.250),
regression khong giam ma con tang (+0.125), va format dau ra dat 1.0000. Tuy nhien
ket qua nay moi duoc do voi `EVAL_LIMIT=8`, nen do tin cay thong ke con mong. Dieu
quan trong nhat em hoc duoc la fine-tuning khong duoc danh gia bang cam giac hoac
final loss. `attn_only` co train loss thap hon `correct` nhung target chi hoa;
`wrong_lr` cho thay chi mot learning rate sai thang la lam target ve 0; `qlora`
tiet kiem VRAM nhung giam target. Don bay that su trong lab nay la to hop mask dung,
prompt/danh gia cong bang, learning-rate scale dung cho LoRA, va viec gan adapter
vao text-linear voi ngan sach tham so duoc kiem soat. Neu can quyet dinh production,
em se rerun full eval khong gioi han mau, tang so qualitative cases, va bo sung
monitoring format/truncation.

**Ba dieu toi hoc duoc**:
1. Mask proof la buoc bat buoc: neu cau hoi nam trong loss thi cac chi so loss ve sau
   co the dep nhung pipeline sai tu goc.
2. Prompt baseline manh la doi thu that su cua fine-tune; fine-tune chi co y nghia
   khi thang duoc baseline (b), khong phai baseline naive.
3. Final loss khong phai metric quyet dinh. Target score, regression, format va
   latency moi cho biet adapter co dang de dung hay khong.

**Neu co them 2 gio nua, toi se thu:** rerun full eval khong `EVAL_LIMIT`, sau do
chay them NB6 merge/hot-swap de kiem tra adapter sau merge co giu diem khong. Em
cung se tang `max_new_tokens`/audit stopping token cho cac ca JSON bi cat.

---

## Phu luc - thuong da lam

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset mien rieng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kem `valid_trace_rate`)
- [ ] B4 quet rank co kiem soat
- [ ] B5 HuggingFace Hub - link:
