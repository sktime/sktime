import os
import sys
from github import Github

def create_workflow_failure_issue(token, run_id):
    try:
        # Initialize GitHub client
        g = Github(token)
        repo = g.get_repo(os.environ['GITHUB_REPOSITORY'])
        
        # Get workflow run details
        workflow_run = repo.get_workflow_run(int(run_id))
        
        # Create issue title and body
        title = f"Workflow Failure: {workflow_run.name}"
        body = f"""
Workflow failure detected in {workflow_run.name}

- Workflow: [{workflow_run.name}]({workflow_run.html_url})
- Run ID: {run_id}
- Triggered by: {workflow_run.triggering_actor.login}
- Status: {workflow_run.conclusion}

Please investigate this failure and fix the underlying issues.
        """
        
        # Create issue with label
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=['good first issue']
        )
        
        print(f"Created issue #{issue.number}")
        return True
        
    except Exception as e:
        print(f"Error creating issue: {str(e)}")
        return False

if __name__ == "__main__":
    token = os.environ['GITHUB_TOKEN']
    run_id = os.environ['WORKFLOW_RUN_ID']
    
    success = create_workflow_failure_issue(token, run_id)
    if not success:
        sys.exit(1)