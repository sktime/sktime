import os
import sys
from github import Github

def comment_on_failures(token, pr_number):
    try:
        # Initialize GitHub client
        g = Github(token)
        repo = g.get_repo(os.environ['GITHUB_REPOSITORY'])
        pr = repo.get_pull(int(pr_number))
        
        # Get workflow runs for this PR
        workflow_runs = repo.get_workflow_runs(
            branch=pr.head.ref,
            event='pull_request'
        )
        
        # Collect failed runs
        failed_runs = []
        for run in workflow_runs:
            if run.conclusion == 'failure':
                failed_runs.append({
                    'name': run.name,
                    'url': run.html_url,
                    'conclusion': run.conclusion
                })
        
        if not failed_runs:
            print("No failed runs found")
            return True
            
        # Create failure summary
        comment_body = "## CI Failure Summary\n\nThe following CI jobs have failed:\n\n"
        for run in failed_runs:
            comment_body += f"- [{run['name']}]({run['url']}) - Status: {run['conclusion']}\n"
        
        comment_body += "\nPlease check the logs and fix the failing tests."
        
        # Add or update comment
        comments = pr.get_issue_comments()
        failure_comment_exists = False
        
        for comment in comments:
            if "CI Failure Summary" in comment.body:
                comment.edit(comment_body)
                failure_comment_exists = True
                break
                
        if not failure_comment_exists:
            pr.create_issue_comment(comment_body)
            
        return True
        
    except Exception as e:
        print(f"Error commenting on PR: {str(e)}")
        return False

if __name__ == "__main__":
    token = os.environ['GITHUB_TOKEN']
    pr_number = os.environ['PR_NUMBER']
    
    success = comment_on_failures(token, pr_number)
    if not success:
        sys.exit(1)