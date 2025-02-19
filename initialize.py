import os
from util import process_and_store_documents


file_paths = {
    "Quy trình": ["/Users/vuthanhdat/deepseek_rag_llm_chatbot/75_QT_QLVHKT_cac_he_thong_dich_vu_so_hop_tac_CSP_lan_3.pdf", "/Users/vuthanhdat/deepseek_rag_llm_chatbot/76_QT_QLVHKT_he_thong_dv_so_MobiFone_lan_3.pdf"],
    "Quyết định": ["/Users/vuthanhdat/deepseek_rag_llm_chatbot/1470_QĐ-MOBIFONE_Bộ_tiêu_chuẩn_ATTT.pdf", "/Users/vuthanhdat/deepseek_rag_llm_chatbot/1528_QĐ-MOBIFONE_Quy_chế_QLVHKT_mạng_VT_CNTT_2024.pdf"]
}


for group, paths in file_paths.items():
    save_path = f"./chroma_db/{group.lower().replace(' ', '_')}"
    if not os.path.exists(save_path):  
        print(f"Đang xử lý tài liệu nhóm {group}...")
        process_and_store_documents(paths, save_path)
        print(f"Đã lưu vector cho nhóm {group}")
    else:
        print(f"Vector của nhóm {group} đã tồn tại, bỏ qua xử lý.")
