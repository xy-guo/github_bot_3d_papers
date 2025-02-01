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
os.environ['https_proxy'] = os.environ["PROXY"]

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
RESEARCH_AREAS = ['3D reconstruction', 'Mesh Reconstruction', '3D generation', "Multi-view Stereo", "Autonomous Driving", "Video Generation"]  # Your research areas

# OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Replace with your actual OpenAI API Key

# 163 Mail SMTP configuration
SMTP_SERVER = "smtp.163.com"                    # 163 Mail SMTP server
SMTP_PORT = 465                                 # 163 SMTP SSL port
SENDER_EMAIL = os.environ["SENDER_EMAIL"]         # Your 163 email address
SENDER_PASSWORD = os.environ["SENDER_PASSWORD"]  # 163 Mail authorization code (NOT your login password)
RECEIVER_EMAIL = os.environ["RECEIVER_EMAIL"]         # Replace with the recipient's email


if not all([OPENAI_API_KEY, SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
    raise ValueError("Please configure the OpenAI API Key, 163 Mail SMTP settings, and email addresses.")

# ========== Step 2: Helper Functions ==========

def fetch_arxiv_papers(search_query="cat:cs.CV", max_results=100):
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
    relate_score: float


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
            research_topic: str,  # main research topic with chinese translation (e.g., 3D reconstruction 三维重建)
            keywords: List[str],  # paper keywords with chinese translation
            contributions: List[str],  # key contributions and novelty, listed in items, each item is a string with chinese translation
            approach: List[str],  # algorithm input --> step1 --> step2 --> step3 --> algorithm output, each step is a string with chinese translation, e.g. input: key frames 关键帧
            relate_score: float  # relevance score (0-10)
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
    print(f"Sending email to {RECEIVER_EMAIL}\n{subject}\n{content}")

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
    not_related_papers = []
    for paper in latest_papers:
        title = paper.title
        abstract = paper.summary
        print(f"Analyzing paper: {title}")
        res: PaperSummary = ask_gpt_if_3d_relevant(title, abstract)
        if res.is_related:
            selected_papers.append((res, paper))
            print(f"\t{res.research_topic}: {res.keywords}")
            print(f"\t{res.approach}")
            if os.environ.get("DEBUG") == "1":
                if len(latest_papers) > 1 and len(not_related_papers) > 1:
                    break
        else:
            print(f"\t[Not Related]")
            not_related_papers.append((res, paper))
    print("\n\n")

    # If there are papers that match the condition, assemble and send an email
    date = selected_papers[0][1].updated.date()
    email_subject = f"Daily Arxiv Papers {date}"
    lines = []

    if selected_papers:
        selected_papers.sort(key=lambda x: x[0].relate_score, reverse=True)
        for idx, (summary, paper) in enumerate(selected_papers):
            paper_id = paper.entry_id.split('/')[-1]
            title = paper.title
            link = paper.entry_id
            topic = summary.research_topic
            keywords = summary.keywords
            contributions = summary.contributions
            approach = summary.approach
            score = summary.relate_score

            lines.append(f"[Paper {idx + 1}, r{score}]: {topic} - {paper_id} - {title}")
            lines.append(f"URL: {link}")
            lines.append(f"关键字: {', '.join(keywords)}")

            lines.append("\n")
            lines.append("Pipeline:")
            for i, step in enumerate(approach, 1):
                lines.append(f"{i}. {step}")
            
            lines.append("\n")
            lines.append("Contributions:")
            for i, contrib in enumerate(contributions, 1):
                lines.append(f"{i}. {contrib}")
            lines.append("-----------------\n")

    print("\n\nNot related papers:\n\n")
    if not_related_papers:
        not_related_papers.sort(key=lambda x: x[0].relate_score, reverse=True)
        for idx, (summary, paper) in enumerate(not_related_papers):
            paper_id = paper.entry_id.split('/')[-1]
            title = paper.title
            link = paper.entry_id
            topic = summary.research_topic
            keywords = summary.keywords
            approach = summary.approach
            score = summary.relate_score

            lines.append(f"[Others {idx + 1}, r{score}]: {topic} - {paper_id} - {title}")
            lines.append(f"URL: {link}")
            lines.append(f"关键字: {', '.join(keywords)}")
            
            lines.append("\n")
            lines.append("Pipeline:")
            for i, step in enumerate(approach, 1):
                lines.append(f"{i}. {step}")
            
            lines.append("\n")
            lines.append("Contributions:")
            for i, contrib in enumerate(contributions, 1):
                lines.append(f"{i}. {contrib}")
            lines.append("-----------------\n")

    email_content = "\n".join(lines)
    send_email(email_subject, email_content)

if __name__ == "__main__":
    main()