import os
import sys
from github import Github

def is_doc_file(filename):
    return filename.endswith('.md') or filename.startswith('docs/')

def comment_on_doc_changes(token, pr_number):
    try:
        # Initialize GitHub client
        g = Github(token)
        repo = g.get_repo(os.environ['GITHUB_REPOSITORY'])
        pr = repo.get_pull(int(pr_number))
        
        # Get changed files
        changed_docs = [f for f in pr.get_files() if is_doc_file(f.filename)]
        
        if not changed_docs:
            print("No documentation changes found")
            return True
            
        # Create comment body
        comment_body = "## Documentation Changes Detected\n\nThe following documentation files were modified:\n\n"
        for file in changed_docs:
            comment_body += f"- [{file.filename}]({file.raw_url})\n"
            
        # Add or update comment
        comments = pr.get_issue_comments()
        doc_comment_exists = False
        
        for comment in comments:
            if "Documentation Changes Detected" in comment.body:
                comment.edit(comment_body)
                doc_comment_exists = True
                break
                
        if not doc_comment_exists:
            pr.create_issue_comment(comment_body)
            
        return True
        
    except Exception as e:
        print(f"Error commenting on PR: {str(e)}")
        return False

if __name__ == "__main__":
    token = os.environ['GITHUB_TOKEN']
    pr_number = os.environ['PR_NUMBER']
    
    success = comment_on_doc_changes(token, pr_number)
    if not success:
        sys.exit(1)