import os
from rag.database import retrieve_text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def get_context_string(question: str, k: int = 4) -> str:
    """
    Queries the vector database for relevant documents and creates a context string.

    Args:
        question: The user's question.
        k: The number of documents to retrieve.

    Returns:
        A string containing the formatted context.
    """
    retrieved_docs = retrieve_text(question, k=k)

    context = ""
    for doc in retrieved_docs:
        cleaned_content = doc.page_content.encode("utf-8", "ignore").decode(
            "utf-8", "ignore"
        )
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context += f"Source: {source}, Page: {page}\n"
        context += cleaned_content + "\n\n"

    return context


def get_gemini_response(question: str, context: str, chat_history: list) -> str:
    """
    Calls the Gemini LLM using LangChain to get an answer based on the question, context, and chat history.

    Args:
        question: The user's current question.
        context: The context retrieved from the vector database.
        chat_history: A list of past messages.

    Returns:
        The LLM's response as a string.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        return "API key not configured. Please set the GOOGLE_API_KEY environment variable."

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

    system_prompt = "You are a helpful assistant for National Institute of Technology, Warangal (NITW). Answer the user's questions. If the information is not in the context provided in the user's question, say that you don't have enough information."

    user_prompt = f"""Use the following context to answer my question.
    
Context:
{context}

Question: {question}
"""

    messages = (
        [SystemMessage(content=system_prompt)]
        + chat_history
        + [HumanMessage(content=user_prompt)]
    )

    try:
        response = llm.invoke(messages)
        return response.content, user_prompt
    except Exception as e:
        return f"An error occurred: {e}", user_prompt


if __name__ == "__main__":
    chat_history = []

    print("Welcome to the NITW Chatbot! Type 'exit' to end the conversation.")

    while True:
        user_question = input("\nQ: ")
        if user_question.lower() == "exit":
            break

        context = get_context_string(user_question, k=5)

        answer, user_prompt = get_gemini_response(user_question, context, chat_history)

        print("\nA:", answer)

        chat_history.append(HumanMessage(content=user_prompt))
        # print(chat_history)
        chat_history.append(AIMessage(content=answer))