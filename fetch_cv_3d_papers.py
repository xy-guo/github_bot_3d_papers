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
os.environ['https_proxy'] = os.environ.get("PROXY", "")

import datetime
import arxiv
import feedparser
import openai
import smtplib
import ssl
from typing import List, Union, Tuple, Dict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel
import jinja2
import markdown
from markdown.extensions.tables import TableExtension


# ========== Step 1: Configuration ==========

# Config
RESEARCH_AREAS = {
    "3D Reconstruction and Modeling": [
        "3D Reconstruction 三维重建",
        "Mesh Reconstruction 网格重建",
        "3D Generation 三维生成",
        "Surface Reconstruction 表面重建",
        "Shape Optimization 形状优化",
        "Volumetric Reconstruction 体积重建",
        "Texture Mapping 纹理映射",
        "Point Cloud Processing 点云处理",
        "Depth Estimation 深度估计",
        "Structure from Motion (SfM) 运动结构估计",
        "Photogrammetry 摄影测量",
        "Shape Completion 形状补全",
        "Model Simplification 模型简化",
        "Sparse and Dense Reconstruction 稀疏与密集重建",
        "3D Morphable Models 三维可变形模型"
    ],
    "Image and Video Generation": [
        "Image Generation 图像生成",
        "Video Generation 视频生成",
        "Diffusion Models 扩散模型",
        "ControlNet 控制网络",
        "Generative Adversarial Networks (GANs) 生成对抗网络",
        "Neural Style Transfer 神经风格迁移",
        "Image Synthesis 图像合成",
        "Super-Resolution 超分辨率",
        "Image Inpainting 图像修复",
        "Video Prediction 视频预测",
        "Frame Interpolation 帧插补",
        "Conditional Generation 条件生成",
        "Unconditional Generation 无条件生成",
    ],
    "Autonomous Systems and Robotics": [
        "Autonomous Driving 自动驾驶",
        "Robotic Perception 机器人感知",
        "Visual Odometry 视觉里程计",
        "Simultaneous Localization and Mapping (SLAM) 同时定位与地图构建",
        "Path Planning 路径规划",
        "Sensor Fusion 传感器融合",
        "Autonomous Navigation 自主导航",
        "Obstacle Detection 障碍物检测",
        "Behavior Prediction 行为预测",
        "Localization 定位",
        "Mapping 地图构建",
        "Reinforcement Learning in Robotics 强化学习在机器人中的应用"
    ],
    "Multi-view and Stereo Vision": [
        "Multi-view Stereo 多视角立体",
        "Stereo Matching 立体匹配",
        "Stereo Vision 立体视觉",
        "Multi-view Geometry 多视图几何",
        "Epipolar Geometry 极线几何",
        "Disparity Estimation 视差估计",
        "Multi-camera Systems 多摄像头系统",
        "Depth Map Fusion 深度图融合",
        "Multi-view Consistency 多视角一致性",
        "Novel View Synthesis 视图合成",
    ],
    "Neural Rendering": [
        "3D Gaussian 三维高斯",
        "Gaussian Splatting 高斯点云",
        "Neural Rendering 神经渲染",
        "Point Cloud Processing 点云处理",
        "Neural Radiance Fields (NeRF) 神经辐射场",
        "Mesh-Based Rendering 基于网格的渲染",
        "Volume Rendering 体积渲染",
        "Differentiable Rendering 可微渲染",
        "Appearance Modeling 外观建模",
        "Lighting Estimation 光照估计",
        "Material Property Estimation 材料属性估计",
        "Real-time Rendering 实时渲染",
        "Photorealistic Rendering 照相真实渲染",
        "Generative Neural Rendering 生成式神经渲染",
        "Hybrid Rendering 混合渲染"
    ],
    "VLM & VLA": [
        "Vision-Language Models (VLMs) 视觉语言模型",
        "Vision-Language Alignment (VLA) 视觉语言对齐",
        "Large Language Models (LLMs) 大型语言模型",
        "Multimodal Learning 多模态学习",
        "Cross-modal Retrieval 跨模态检索",
        "Visual Question Answering (VQA) 视觉问答",
        "Image Captioning 图像描述生成",
        "Text-to-Image Generation 文本到图像生成",
        "Image-Text Matching 图像文本匹配",
        "Multimodal Embedding 多模态嵌入",
        "Visual Commonsense Reasoning 视觉常识推理",
        "Contextual Reasoning 上下文推理",
        "Zero-shot Learning 零样本学习",
        "Few-shot Learning 少样本学习"
    ]
}

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

def fetch_arxiv_papers(search_query="cat:cs.CV", max_results=200):
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
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    for result in client.results(search):
        results.append(result)
    return results

def fetch_arxiv_papers_rss(search_query="cat:cs.CV", max_results=200):
    # ArXiv cs.CV RSS feed URL
    rss_url = "https://arxiv.org/rss/cs.CV"

    # 解析 RSS feed
    feed = feedparser.parse(rss_url)

    # 遍历并打印论文信息
    papers = []
    for i, entry in enumerate(feed.entries):
        title = entry.title
        link = entry.link
        summary = entry.summary
        tags = [tag.term for tag in entry.tags]
        date = entry.published
        date = datetime.datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
        authors = entry.authors
        if not "Announce Type: new" in summary:
            continue

        print(f"Paper {i+1}/{len(feed.entries)}")
        print(f"Title: {title}")
        print(f"Link: {link}")
        print(f"Date: {date}")
        print(f"Authors: {authors}")
        print(f"Tags: {tags}")
        print(f"Abstract: {summary}")
        print("\n")

        papers.append(
            arxiv.Result(
                title=title,
                summary=summary,
                updated=date,
                entry_id=link,
                authors=authors,
            )
        )
    return papers

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
    score_reason: str


def ask_gpt_if_3d_relevant(title: str, abstract: str, authors: List) -> bool:
    """
    Use ChatGPT API to check whether the paper is related to 
    '3D Reconstruction' or '3D Generation'. 
    Returns True if relevant, otherwise False.
    """
    system_prompt = (
        "You are a PhD researcher in computer vision tasked with filtering research papers relevant to your areas of study. "
        f"Your research areas include: {RESEARCH_AREAS}."
    )

    prompt = (
        "You will be provided with a paper title and abstract. "
        "Determine if the paper is related to your research areas and respond using the strict JSON format below:\n\n"
        "```json\n"
        "{\n"
        '  "is_related": true,\n'
        '  "research_topic": "3D reconstruction 三维重建",\n'
        '  "keywords": ["3D reconstruction", "autonomous driving"],\n'
        '  "contributions": [\n'
        '    "Developed novel algorithms 提出了新算法",\n'
        '    "Integrated multi-view images 与多视角图像集成"\n'
        '  ],\n'
        '  "approach": [\n'
        '    "Input: Multi-view images 多视角图像",\n'
        '    "Step1: Data integration 数据集成",\n'
        '    "Step2: Algorithm development 算法开发",\n'
        '    "Step3: Model evaluation 模型评估",\n'
        '    "Output: Enhanced 3D models 改进的三维模型"\n'
        '  ],\n'
        '  "relate_score": 9.0,\n'
        '  "score_reason": "The paper presents a novel approach to 3D reconstruction and has high relevance to my research areas."\n'
        "}\n"
        "```\n\n"
        "### Instructions:\n"
        "- `research_topic` should include the primary English term followed by its Chinese translation, separated by a space.\n"
        "- **Focus on broad and general topics** within each research area to capture foundational and widely applicable research, rather than highly specialized subfields.\n"
        "- **Assess the quality** of the research based on its contribution, innovation, methodology, and potential impact. Assign a higher `relate_score` (closer to 10) to high-quality, impactful papers and a lower score to those of lesser quality or relevance.\n"
        "- Ensure `relate_score` is a float between 0 and 10, representing the relevance score to your research areas. High-quality research should receive a higher score.\n"
        "- In the `contributions` section, highlight the main contributions and innovations of the paper that demonstrate its research value.\n"
        "- In the `approach` section, provide a succinct overview of the methodology steps, emphasizing their general applicability and significance.\n"
        "- **Do not include any additional text or comments** outside the specified JSON format.\n"
        "- Maintain **consistency** in the bilingual format, ensuring each English term is accurately paired with its Chinese translation.\n\n"
        f"**Paper Title**: {title}\n\n"
        f"**Abstract**: {abstract}\n\n"
        f"**Authors**: {authors}\n\n"
        "Please generate the JSON response accordingly."
    )

    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.feidaapi.com/v1/" if len(OPENAI_API_KEY) <= 51 else None)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # If you have GPT-4 access, you can switch to gpt-4
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
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
    part = MIMEText(content, "html", "utf-8")
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
    papers = fetch_arxiv_papers_rss()
    
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
        authors = paper.authors
        print(f"Analyzing paper: {title}")
        res: PaperSummary = ask_gpt_if_3d_relevant(title, abstract, authors)
        if res.is_related:
            selected_papers.append((res, paper))
            print("\t", res)
            if os.environ.get("DEBUG") == "1":
                if len(selected_papers) > 2 and len(not_related_papers) > 2:
                    break
        else:
            print(f"\t[Not Related]")
            not_related_papers.append((res, paper))
    print("\n\n")

    # If there are papers that match the condition, assemble and send an email
    date = selected_papers[0][1].updated.date()
    email_subject = f"Daily Arxiv Papers {date}"

    selected_papers.sort(key=lambda x: x[0].relate_score, reverse=True)
    not_related_papers.sort(key=lambda x: x[0].relate_score, reverse=True)

    # load template.html and render
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "template.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    def to_jinja2_format(papers: List[Tuple[PaperSummary, arxiv.Result]]) -> List[Dict]:
        outputs = []
        for summary, paper in papers:
            arxiv_id = paper.entry_id.split("/")[-1]
            contributions = [f"{i+1}. {c}" for i, c in enumerate(summary.contributions)]
            output = {
                "title": f"[{summary.relate_score}] {arxiv_id} {paper.title}",
                "score": summary.relate_score,
                "summary": paper.summary,
                "research_topic": summary.research_topic,
                "keywords": summary.keywords,
                "contributions": contributions,
                "pipeline": summary.approach,
                "url": paper.entry_id,
                "authors": paper.authors,
            }
            outputs.append(output)
        return outputs
    
    email_content = template.render(
        date=date,
        related_papers=to_jinja2_format(selected_papers),
        not_related_papers=to_jinja2_format(not_related_papers),
    )

    with open("email.html", "w") as f:
        f.write(email_content)

    send_email(email_subject, email_content)

    # update README.md
    update_readme(to_jinja2_format(selected_papers), date)

def update_readme(papers, date):
    date_str = date.strftime("%Y-%m-%d")
    new_section = f"## Arxiv {date_str}\n\n"
    new_section += "Relavance | Title | Research Topic | Keywords | Pipeline\n"
    new_section += "|------|---------------|----------------|----------|---------|\n"

    def to_md(items):
        return "<br>".join(items)

    for paper in papers:
        new_section += f"{paper['score']} | [{paper['title']}]({paper['url']}) <br> {paper['authors']} | {paper['research_topic']} | {to_md(paper['keywords'])} | {to_md(paper['pipeline'])} |\n"
    new_section += "\n\n"

    # Read the existing README.md
    with open('README.md', 'r') as file:
        content = file.readlines()

    # Find the index of the last date section to replace it
    for index, line in enumerate(content):
        if line.startswith("# Paper List"):
            last_date_index = index + 1
            break

    # Update the README content
    content = content[:last_date_index] + [new_section] + content[last_date_index:]

    # Write the updated content back to README.md
    with open('README.md', 'w') as file:
        file.writelines(content)


if __name__ == "__main__":
    main()