# Daily Updates on 3D-Related Papers

This repository automatically fetches new or updated arXiv papers in the [cs.CV] category every day, checks if they are relevant to "3D reconstruction" or "3D generation" via ChatGPT, and lists them below.

## How It Works
1. A GitHub Actions workflow runs daily at 09:00 UTC.  
2. It uses the script [fetch_cv_3d_papers.py](fetch_cv_3d_papers.py) to:  
   - Retrieve the latest arXiv papers in cs.CV.  
   - Use ChatGPT to filter out those related to 3D reconstruction/generation.  
   - Update this README.md with the new findings.  
   - Send an email via 163 Mail if any relevant papers are found.  

# Paper List
## Newly Found Papers on ...
(Older entries get replaced automatically when the script runs again.)