# ImPerSum
Information Modulated Personalized Summarizer

📘 **BEHAVIOR EXTRACTION & TRAINING INSTANCE GENERATION**

📥 INPUT
CSV file with user interaction sequences.

Required columns:

• UserID   → unique user identifier  
• Docs     → stringified Python list of document IDs  
• Action   → stringified Python list of actions (aligned with Docs)

Example:

UserID,Docs,Action  

U1,"['N1','N2']","['click','summ_gen']"

⚙️ PROCESSING LOGIC

① BEHAVIOR GRAPH CONSTRUCTION

• Each interaction is assigned a unique EdgeID (B1, B2, …)

• First interaction:
    
    User ──(action)──▶ Doc₀

• Subsequent interactions:

    Docᵢ₋₁ ──(action)──▶ Docᵢ

② BEHAVIOR LOOKUP TABLE

• Columns:

  EdgeID | Head | Relation | Tail | User

• Relations:
  
  { click, skip, gen_summ, summ_gen }

③ DWELL TIME AUGMENTATION

• click      → pens dataset dwell ∈ [20, 1230]
• otherwise  → NaN

④ TRAINING INSTANCE EXTRACTION

• For every `summ_gen` event:
  Bhist = all EdgeIDs before this event
  Bpos  = EdgeID of the current summ_gen

• One training instance per summ_gen

📤 OUTPUT

① Behavior Vocabulary (Behavior_Vocab.csv)

• Global behavior graph

• One row per interaction edge

② Training Dataset (train_df)

• Columns:
  UserID | Bhist | Bpos

• Supervision format:
  Bhist ──▶ Bpos

🔍 VALIDATION

All Bpos values are verified to exist in the behavior lookup table.

🧩 USE CASES

• Sequential recommendation  
• Next-behavior prediction  
• Behavior-to-Summary (B2S) modeling  
• User behavior graph learning

📦 DEPENDENCIES
pip install pandas numpy tqdm

▶ RUN

Update CSV path and execute:
behavior_exptraction.ipynb notebook


📘 **CORPUS TO EMBEDDING CONVERSION**

🔹 This script generates dense semantic embeddings for news headlines, news bodies, and summaries using the **E5 (intfloat/e5-base-v2)** transformer model with GPU support.

🔹 Texts are batch-encoded, mean-pooled, L2-normalized, and stored as **NewsID → embedding** mappings in pickle files for efficient downstream retrieval.

🔹 The resulting embeddings are used as fixed representations in personalization, retrieval, and behavior-to-summary modeling pipelines.

▶ RUN

Update pkl path and execute:
embedding_generation.ipynb notebook
