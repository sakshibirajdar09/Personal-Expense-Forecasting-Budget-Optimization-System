import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------------------------
# 🧠 Function: Chat-based assistant
# ---------------------------------
def chat_with_ai(user_query, df):
    """
    Returns AI-generated financial insights based on uploaded data and user question.
    """
    # Summarize data briefly to feed into model
    summary = df.groupby('category')['amount'].sum().reset_index().to_string(index=False)

    prompt_template = PromptTemplate(
        input_variables=["summary", "question"],
        template=(
            "You are a financial assistant. "
            "Here is the user's expense summary:\n{summary}\n\n"
            "Answer this question clearly and helpfully:\n{question}"
        )
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    chain = LLMChain(prompt=prompt_template, llm=llm)

    try:
        response = chain.run({"summary": summary, "question": user_query})
        return response.strip()
    except Exception as e:
        return f"⚠️ Sorry, I couldn’t process your question. ({e})"
