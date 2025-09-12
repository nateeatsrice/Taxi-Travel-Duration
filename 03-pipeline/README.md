## Turning pipeline into a script

After the pipeline is ready in a script (see [.py file](duration-prediction.py)), it is possible to call the pipeline via terminal by:
1. Starting MLFlow: `uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --default-artifact-root ./artifacts`
2. Change directory: `cd 03-deployment/`
2. Running the script: `uv run python duration-prediction.py --year 2024 --month 3`

It will create a model and run it with MLFlow + save the run ID in `run_id.txt`.

Next steps to create a pipeline with Prefect:

## Prefect

### Installation

`uv pip install prefect`

### Basic workflow

Start a local server:
* `uvx prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api`
* `uvx prefect server start`

Then it is possible to go to the dashboard at http://127.0.0.1:4200/dashboard.

Assuming that MLFlow is also running, it is now possible to run the script and observe the results in Prefect as well as in the console.

### Deployment

`prefect project init` - this is not working, the correct command is `uv run prefect init`. This initializes a Prefect project in the current directory.

Go to Prefect UI and create a new work pool. The type to select is "Process", since it's the most basic option.

Deploy the flow with the following command: `uv run prefect deploy duration-prediction.py:run -n nyc-taxi-flow -p "Work Pool 1"`.

-n specifies the deployment name, -p specifies the work pool where the deployment should run

A work pool is basically the execution environment (workers or agents) that listen for deployments

To execute flow runs, we'll need to start a worker: `uv run prefect worker start --pool "Work Pool 1"`.

And then it is possible to run the deployment either from the command line (`prefect deployment run) or from the UI.

**VERY IMPORTANT**: since Prefect involves git and clones a copy of the repo with a current state from the remote, it is very important to first `git push` all the changes!

### Scheduling

Scheduling is easily available from the UI, and there are CLI commands as well.