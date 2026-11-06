# Chinese Tone Scorers：Open WebUI 串接

四個 Workbook3 語氣分類器，輸出 **溫和、中性、不滿、酸**。提供 OpenAI-compatible API，可在 Open WebUI 模型選單選用，或作為聊天模型的評分工具。

| API 模型 ID | 預訓練模型 | 載入與微調方式 | 分類頭 |
| --- | --- | --- | --- |
| `roberta-base` | `hfl/chinese-roberta-wwm-ext` | BF16 + LoRA | CLS pooler → Linear(768, 4) |
| `roberta-large` | `hfl/chinese-roberta-wwm-ext-large` | BF16 + LoRA | CLS pooler → Linear(1024, 4) |
| `llama-8b-lora` | `unsloth/Meta-Llama-3.1-8B` | BF16 + LoRA | 最後一個非 padding token → Linear(4096, 4) |
| `llama-8b-4bit-lora` | 同上、相同 revision | NF4 double quantization + BF16 compute + LoRA | 同上 |

四個 scorer 均取自 `06_workbook3_model_comparison` 的 **seed 3407、完整 496 筆 train 訓練、3 epochs、372 updates** 模型。選擇規則一致，未依 test 分數挑選 seed。這四個 adapter 包含 LoRA 與訓練好的分類頭；五次平均是評估統計量，沒有對應的單一「平均模型」。完整版本、指標與 SHA-256 見 [models.json](models.json)。

## 1. 安裝與下載

本 repository 為私人。先安裝 [GitHub CLI](https://cli.github.com/)，在使用模型的機器登入並取得程式：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh repo clone uurspecial/tone-scorers
cd tone-scorers
```

需要 Python 3.11。建議使用支援 BF16 的 NVIDIA GPU；RTX 5090 使用相容的 CUDA 12.8 PyTorch build。依需要先參照 [PyTorch 安裝頁](https://pytorch.org/get-started/locally/) 安裝正確 GPU 版本，再安裝依賴及下載 adapter：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py --repo uurspecial/tone-scorers
```

可加 `--model roberta-base` 只下載一個模型。權重放在 Release `v1.0.0`，下載後會核對壓縮檔及逐檔 SHA-256。兩個 Llama adapter 超過 GitHub 一般 Git 的 100 MiB 限制，因此不把權重加入 Git 歷史。

首次推論還會下載 `models.json` 固定 revision 的預訓練主體。主體權重未包含在 Release 中；若來源要求授權，先完成來源授權，再設定 `HF_TOKEN`。亦可用 `HF_HUB_CACHE` 指向既有快取。HFL 來源標示 Apache-2.0，Llama 遵循其 Llama 3.1 Community License；詳見各來源模型頁面。

## 2. 啟動 scorer API

在 repository 根目錄執行：

```bash
export SCORER_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SCORER_DEVICE=cuda
uvicorn server:app --host 127.0.0.1 --port 8001 --workers 1
```

將 `SCORER_API_KEY` 保存到你的密碼管理器，並填入 Open WebUI 的 API key 欄位；不要上傳到 GitHub。`.env.example` 僅為設定範例，程式不會自動載入 `.env`。若 Open WebUI 在其他機器／容器中，可依網路配置改為 `--host 0.0.0.0`，並確保服務可被 Open WebUI 後端連到。

服務按需載入，一次只在 GPU 保留一個 scorer，切換模型會卸載前一個。首次下載與模型切換較慢；不要開多個 uvicorn workers。BF16 Llama 需要較多空閒顯存，請先檢查 GPU。`SCORER_DEVICE=cpu` 支援 HFL 模型的 FP32 推論；要對齊原實驗 BF16 設定，請使用 GPU。

```bash
curl http://127.0.0.1:8001/v1/models \
  -H "Authorization: Bearer $SCORER_API_KEY"

curl http://127.0.0.1:8001/v1/score \
  -H "Authorization: Bearer $SCORER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"roberta-base","text":"謝謝你的協助，我會再確認一次。"}'
```

`POST /v1/score` 回傳 `label_id`（1–4）、`label`、四類 `probabilities`、原始 `logits` 與 `input_tokens`。機率由 softmax 計算，未另外校準。文字截斷上限為 512 tokens，與訓練相同。

## 3. 在 Open WebUI 選擇四個 scorer

1. 以管理員開啟 **Admin Panel → Settings → Connections → OpenAI-compatible**，新增連線。
2. URL 填 `http://SCORER_HOST:8001/v1`，Key 填上述 `SCORER_API_KEY`。
3. 儲存並更新模型列表，即可選擇四個模型 ID。
4. 直接輸入要分類的原句，會顯示預測語氣與四類機率。

若 Open WebUI 跑在 Docker，`127.0.0.1` 指向容器本身。可使用可連到主機的 IP，或配置 `host.docker.internal`（Linux Docker 通常需 `--add-host=host.docker.internal:host-gateway`）。scorer 服務需監聽可被容器連到的介面。

分類器每次只評分**最後一則使用者訊息**，不使用先前對話或 system prompt。它輸出分類結果，沒有聊天生成能力；建議關閉這個連線模型的自動標題、標籤及後續問題生成工作，或將這些工作交給其他聊天模型。`temperature`、生成 token 上限等生成參數不影響分類。串流與非串流均支援，串流會在分類完成後一次送出結果。

連線方式參考 [Open WebUI 官方文件](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/)。

## 4. 讓其他聊天模型呼叫 scorer

若要評分另一個模型生成的文字，可在 **Workspace → Tools** 建立工具，貼入 [openwebui_tool.py](openwebui_tool.py)，並設定 Valves：

- `SCORER_URL`：例如 `http://host.docker.internal:8001`，不含 `/v1`。
- `SCORER_API_KEY`：同一把 scorer API key。

在聊天中啟用此工具，讓支援工具呼叫的模型呼叫 `score_tone(text, model)`。工具可選四個 scorer，將欲評分文字明確傳入。這是按需工具呼叫，不會自動對每個回答執行評分。

## 5. 驗證與重新匯出

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

8 項測試使用替身分類器，驗證 API 驗證、文字選取、四模型列表、串流格式、模型權重／壓縮檔校驗及安全解壓縮。[validation.json](validation.json) 另記錄四個實際模型各 496 筆原始測試資料的推論對照：共 1,984 筆預測與 logits 均與原實驗完全一致。GPU 推論保留原訓練的 BF16 autocast，LoRA 參數保存為 FP32，分類頭為 BF16。單筆輸入路徑也已逐模型驗證；批次大小改變時可能有浮點數值差異。

維護者可從原始實驗重新匯出，原訓練產物不會被修改：

```bash
python prepare_release.py --experiment /path/to/06_workbook3_model_comparison
```

匯出時會將 adapter 的預訓練主體路徑改成可攜的模型 ID、固定 revision，並整理分類頭模組清單；`adapter_model.safetensors` 的內容保持逐位元相同。`adapters/` 與 `dist/` 已加入 `.gitignore`。發布時上傳 `dist/*.tar.gz` 及 `dist/SHA256SUMS` 到同一個 GitHub Release。
