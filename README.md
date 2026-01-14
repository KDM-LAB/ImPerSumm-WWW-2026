# ImPerSum
**Information-Modulated Personalized Summarizer**

This repository contains the complete experimental pipeline for **ImPerSum**, including behavior graph construction, embedding preparation, dimensionality reduction, behavior encoding, and personalized summary generation with latent injection into T5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📘 1. BEHAVIOR EXTRACTION & TRAINING INSTANCE GENERATION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Input

CSV file with user interaction sequences.

Required columns:

• `UserID`   → unique user identifier  
• `Docs`     → stringified Python list of document IDs  
• `Action`   → stringified Python list of actions aligned with `Docs`


### Processing Logic

① **Behavior Graph Construction**  

• Each interaction is mapped to a unique `EdgeID` (B1, B2, …)

User ──(action)──▶ Doc₀
Docᵢ₋₁ ──(action)──▶ Docᵢ


② **Behavior Lookup Table**

EdgeID | Head | Relation | Tail | User | Dwell

③ **Dwell Time Augmentation**  

• `click` → dwell from PENS dataset ∈ [20, 1230]  

• otherwise → NaN

④ **Training Instance Extraction**  
For every `summ_gen` action:

Bhist = all EdgeIDs before the event
Bpos = EdgeID of the current summ_gen


One training instance is created per `summ_gen`.

### Outputs
• `Behavior_Vocab.csv` — global behavior graph  
• `train_df` — supervision tuples `(Bhist → Bpos)`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📘 2. CORPUS → EMBEDDING CONVERSION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dense semantic embeddings are generated for:

• News headlines  
• News bodies  
• Summaries  

Using:

• **E5 / T5 encoders**
• Mean pooling + L2 normalization
• Stored as `{ID → embedding}` pickle dictionaries

These embeddings serve as **fixed semantic nodes** for all downstream modules.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📘 3. PCA DIMENSIONALITY REDUCTION (768 → 192)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To reduce memory and improve efficiency, all embeddings are projected to 192 dimensions.

### Key Properties

• **Incremental PCA** (shared across headline, body, summary embeddings)  
• Batch-wise fitting for scalability  
• Single PCA model reused across all modalities  

### Outputs 

headline_T5_pca192.pkl
newsbody_T5_pca192.pkl
summary_T5_pca192.pkl


Each preserves the original ID → vector mapping with reduced dimensionality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📘 4. BEHAVIOR ENCODER (Paper-Correct)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The **BehaviorEncoder** implements the paper’s formulation with:

• Action-specific gates  
• KDE-based mutual information estimation  
• Short-term, long-term, and event memory kernels  
• Adaptive Memory Fusion (AMF)  
• Tail-ID classification objective  

### Encoder Outputs

• Final user behavior state `z_b`  
• Predicted next-behavior embedding  
• Supervised next-tail prediction loss  

This encoder explicitly models **information modulation per action**, consistent with the ImPerSum formulation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📘 5. PSEUDO-INVERSE SUMMARY NODE RECOVERY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A learned pseudo-inverse mapping recovers a latent **summary node embedding** from the user behavior state:
z_b → ê_b → ê_s


This enables summary-level personalization without direct text conditioning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📘 6. PERSONALIZED GENERATION (B2S MODEL)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The **B2SModel** integrates:

• BehaviorEncoder  
• Pseudo-Inverse Mapper  
• Cross-Attention (Eq. 19) between recovered summary node and document  
• Latent prefix injection into **T5-Large** decoder  

### Training Objective
Total Loss = 0.5 × Behavior Encoding Loss + 0.5 × T5 Generation Loss


### Generation
Personalized summaries are generated **autoregressively** using the injected latent prefix, without modifying the T5 architecture.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📘 7. EVALUATION ARTIFACTS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The evaluation pipeline outputs:
Bpos
true_tail_id
generic summary
predicted personalized summary
gold summary

