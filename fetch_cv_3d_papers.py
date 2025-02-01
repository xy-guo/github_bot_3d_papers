#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script: Use arXiv API + ChatGPT API to automatically fetch new Computer Vision papers 
published (or updated) today from arXiv, filter out those that are relevant to
"3D Reconstruction/3D Generation," and send an email notification via 163 Mail.

Usage:
1. Install required dependencies:
   pip install arxiv
   pip install openai

2. Configure parameters in the code:
   - OPENAI_API_KEY: Replace with your actual OpenAI API Key
   - SMTP settings: Use 163 SMTP, fill in your account and authorization code
3. Run the script:
   python fetch_cv_3d_papers.py
"""

import os
os.environ['https_proxy'] = "http://10.8.202.73:12332"

import datetime
import arxiv
import openai
import smtplib
import ssl
from typing import List, Union, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel


# ========== Step 1: Configuration ==========

# Config
RESEARCH_AREAS = ['3D reconstruction', '3D generation']

# OpenAI API key
OPENAI_API_KEY = "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"  # Replace with your actual OpenAI API Key

# 163 Mail SMTP configuration
SMTP_SERVER = "smtp.163.com"                    # 163 Mail SMTP server
SMTP_PORT = 465                                 # 163 SMTP SSL port
SENDER_EMAIL = "guoxy95@163.com"         # Your 163 email address
SENDER_PASSWORD = "your_163_email_authorization_code"  # 163 Mail authorization code (NOT your login password)
RECEIVER_EMAIL = "receiver@example.com"         # Replace with the recipient's email

# ========== Step 2: Helper Functions ==========

def fetch_arxiv_papers(search_query="cat:cs.CV", max_results=50):
    """
    Fetch a list of papers from arXiv that match the search conditions.
    Default search category: Computer Vision (cs.CV).
    Adjust max_results to control how many papers to retrieve.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = []
    for result in client.results(search):
        results.append(result)
    return results

def is_same_day(date1: datetime.datetime, date2: datetime.datetime) -> bool:
    """
    Check if two datetime objects are on the same calendar day.
    """
    return date1.date() == date2.date()


class PaperSummary(BaseModel):
    is_related: bool
    research_topic: str
    keywords: List[str]
    contributions: List[str]
    approach: List[str]


def ask_gpt_if_3d_relevant(title: str, abstract: str) -> bool:
    """
    Use ChatGPT API to check whether the paper is related to 
    '3D Reconstruction' or '3D Generation'. 
    Returns True if relevant, otherwise False.
    """
    openai.api_key = OPENAI_API_KEY

    system_prompt = (
        "You are a computer vision PhD researcher to filter out computer vision papers related to your research. "
        f"Your research areas are {RESEARCH_AREAS}."
    )

    prompt = (
        "You will receive a paper title and abstract. "
        "Determine if it is related to your research area. "
        f"Paper Title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        "Please answer in json format:"
        """dict(
            is_related: bool,  # is related to your research topic
            research_topic: str,  # main research topic & task
            keywords: List[str],  # paper keywords
            contributions: List[str],  # main contributions, listed in items
            approach: List[str],  # input --> step1 --> step2 --> step3 --> solution
            )
        """
    )

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # If you have GPT-4 access, you can switch to gpt-4
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format=PaperSummary,
    )

    # Get the response from ChatGPT
    answer = completion.choices[0].message.parsed
    return answer

def send_email(subject: str, content: str):
    """
    Send an email via 163 Mail SMTP.
    subject: Email subject
    content: Plain text email body
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    # Attach text content to the MIMEMultipart
    part = MIMEText(content, "plain", "utf-8")
    msg.attach(part)

    # Use SSL to connect to the SMTP server
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("Email has been successfully sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ========== Step 3: Main Logic ==========

def main():
    print("Fetching today's new papers (cs.CV) from arXiv...")
    papers = fetch_arxiv_papers()
    
    # Print basic info for debugging/logging
    for i, paper in enumerate(papers):
        time = paper.updated
        print(f"[{i}] {paper.entry_id.split('/')[-1]} {paper.title}: {paper.summary[:50]}... (Updated: {time})")

    # Filter papers that are updated today
    last_day = papers[0].updated
    latest_papers = [p for p in papers if is_same_day(p.updated, last_day)]
    print(f"Number of papers updated today: {len(latest_papers)}")

    selected_papers = []
    for paper in latest_papers:
        title = paper.title
        abstract = paper.summary
        print(f"Analyzing paper: {title}")
        res: PaperSummary = ask_gpt_if_3d_relevant(title, abstract)
        if res.is_related:
            selected_papers.append((res, paper))
            print(f"\t{res.research_topic}: {res.keywords}")
            print(f"\t{res.approach}")
        else:
            print(f"\t[Not Related]")

    # If there are papers that match the condition, assemble and send an email
    if selected_papers:
        email_subject = "New Papers Related to {}"
        lines = []
        for idx, p in enumerate(selected_papers):
            lines.append(f"{idx}. {p.title}\n")
            lines.append(f"Authors: {', '.join(str(author) for author in p.authors)}\n")
            lines.append(f"Link: {p.entry_id}\n")
            lines.append(f"Abstract: {p.summary}\n")
            lines.append("=" * 80 + "\n")

        email_content = "\n".join(lines)
        send_email(email_subject, email_content)
    else:
        print("No papers related to 3D Reconstruction / 3D Generation today. No email sent.")

if __name__ == "__main__":
    main()