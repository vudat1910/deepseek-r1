import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from util import process_and_store_documents, get_retriever
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate



def get_custom_prompt():
    """Define and return the custom prompt template."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Bạn là một trợ lý về quy trình, hãy đưa ra thông tin chính xác nhất cho người hỏihỏi. Thực hiện theo các nguyên tắc sau:\n"
"1. Trả lời câu hỏi chỉ bằng thông tin từ các tệp PDF đã tải lên.\n"
"2. Sử dụng ngôn ngữ tiếng việt, đơn giản, rõ ràng.\n"
"3. Nếu câu trả lời không có trong tài liệu, hãy nói: 'Tôi không thể tìm thấy thông tin liên quan trong các tài liệu được cung cấp.'\n"
"4. Không suy đoán, giả định hoặc bịa đặt thông tin.\n"
"5. Nếu câu trả lời không có trong tài liệu, tuyệt đối không trả lời thông tin gì nữa.\n"
"6. Duy trì giọng điệu chuyên nghiệp và sắp xếp câu trả lời rõ ràng (ví dụ: gạch đầu dòng, giải thích từng bước).\n"
"7. Khuyến khích các câu hỏi tiếp theo bằng cách hỏi xem có cần làm rõ thêm không.\n"
"8. Cung cấp các ví dụ để làm rõ các khái niệm khi hữu ích.\n"
"9. Giữ câu trả lời ngắn gọn, tập trung và thân thiện với bài kiểm tra.\n"
"10. Đưa ra câu trả lời bằng tiếng việt, không được dùng ngôn ngữ khác."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "Cung cấp câu trả lời chính xác, dễ hiểu và có cấu trúc tốt."
        )
    ])


def initialize_qa_chain(*args, **kwargs):
    if not st.session_state.qa_chain:
        llm = ChatOllama(model="llama3.2:3b", temperature=0.3)

        selected_group = st.session_state.get("selected_group", "")
        if not selected_group:
            st.error("Vui lòng chọn nhóm tài liệu trước khi tiếp tục.")
            return None

        retriever = get_retriever(f"chroma_db/{selected_group.lower().replace(' ', '_')}")
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": get_custom_prompt()}
        )
    return st.session_state.qa_chain







def initialize_session_state():
    """Khởi tạo trạng thái phiên"""
    if "vector_stores" not in st.session_state:
        st.session_state.vector_stores = {}
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_sidebar():
    with st.sidebar:
    
        selected_group = st.radio("Chọn nhóm tài liệu:", ["Quy trình", "Quyết định"])
        st.session_state.selected_group = selected_group




def chat_interface():
    """Giao diện chat"""
    st.title("Chatbot")
    st.markdown("Hỏi đáp quy trình nội bộ")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Bạn muốn hỏi gì?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner("Đang tìm kiếm thông tin"):
                try:
                    qa_chain = initialize_qa_chain(st.session_state.selected_file)

                    if not qa_chain:
                        full_response = "Vui lòng chọn một tài liệu và tải dữ liệu"
                    else:
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response["result"]
                except Exception as e:
                    full_response = f"Lỗi: {str(e)}"

            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    initialize_session_state()
    display_sidebar()
    chat_interface()


if __name__ == "__main__":
    main()
