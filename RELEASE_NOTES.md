四個 Workbook3 語氣 scorer：`roberta-base`、`roberta-large`、`llama-8b-lora`、`llama-8b-4bit-lora`。

全部採用 seed 3407、完整 496 筆 train、3 epochs 的 LoRA 與四分類頭。adapter 權重內容與原實驗逐位元相同；載入設定已改為可攜模型 ID 與固定 revision。預訓練主體權重需另由模型來源取得。

提供 OpenAI-compatible 模型列表、一般與 SSE 串流分類回覆、結構化 `/v1/score`，以及 Open WebUI 工具。服務一次保留一個 GPU 模型，切換時卸載前一個。

驗證：8 項 API／封裝測試通過；四個模型各 496 筆原始測試資料，共 1,984 筆預測與 logits 完全一致。詳見 repository 的 `validation.json`。這些是四個單一模型，不是五折 ensemble 或五次指標平均模型。

下載：登入 GitHub 後執行 `python download_models.py --repo uurspecial/tone-scorers`，會核對各壓縮檔及逐檔 SHA-256。串接步驟見 README。
