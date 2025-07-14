import asyncio
import json

import pandas as pd
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

from dotenv import load_dotenv


load_dotenv()


def load_csv(file_path: str) -> Dict[str, Any]:
    """Load a CSV file and return its records as a list of dictionaries."""
    if file_path:
        df = pd.read_csv(file_path)
    else:
        df = sns.load_dataset("titanic")
    records = df.to_dict(orient='records')
    return {"data": records, "summary": {"shape": df.shape, "columns": df.columns.tolist()}}

LoadCSVTool = FunctionTool(
    func=load_csv,
    name="LoadCSV",
    description="Load a CSV from disk and return its records as a list of dicts. Requires 'file_path'.",
)


def clean_dataframe(file_path: str, actions: List[str]) -> Dict:
    """Clean and preprocess a dataset given as a file path and actions.
    Supported actions include:
    - dropna: Remove rows with any missing values.
    - fillna: Fill missing values with a specified value.
    - encode: Encode categorical variables as integers.
    - normalize: Normalize specified columns to a 0-1 range.
    """
    df = pd.read_csv(file_path)
    if "dropna" in actions:
        df = df.dropna()
    if "fillna" in actions:
        for col in df.columns:
            # Fill NaN values with 0 for numeric columns
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            else:
                # For non-numeric columns, fill with a placeholder or leave as is
                df[col] = df[col].fillna("Unknown")
    if "encode" in actions:
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category').cat.codes
                elif df[col].dtype == 'category':
                    df[col] = df[col].cat.codes
                elif df[col].dtype == 'string':
                    df[col] = df[col].astype('category').cat.codes
                else:
                    continue
            except Exception as e:
                print(f"Encoding error for column {col}: {e}")
                continue

    if "normalize" in actions:
        for col in df.columns:
            try:
                min_val, max_val = df[col].min(), df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            except TypeError:
                # Skip non-numeric columns
                continue
    if "remove_duplicates" in actions:
        df = df.drop_duplicates()

    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head().to_dict(orient="records"),
    }
    output_file_path = file_path.replace(".csv", "_cleaned.csv")
    df.to_csv(output_file_path, index=False)
    return {
        "cleaned_data": df.to_dict(orient="records"),
        "summary": summary,
        "origin_file_path": file_path,
        "actions": actions,
        "output_file_path": output_file_path
    }


PandasDataPreparationTool = FunctionTool(
    func=clean_dataframe,
    name="CleanDataFrame",
    description="""
    Clean and preprocess a dataset given as a file path and actions.
    Supported actions include:
    - dropna: Remove rows with any missing values.
    - fillna: Fill missing values with a specified value.
    - encode: Encode categorical variables as integers.
    - normalize: Normalize specified columns to a 0-1 range.
    """
)



def perform_eda(file_path: str) -> Dict[str, Any]:
    """
    Perform exploratory data analysis (EDA) on a dataset given as a file path.
    Returns summary statistics, missing value information, and a base64-encoded correlation heatmap.
    """
    df = pd.read_csv(file_path)
    stats = df.describe(include='all').to_dict()
    missing = df.isnull().sum().to_dict()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

    # save
    heatmap_file_path = file_path.replace(".csv", "_correlation_heatmap.png")
    with open(heatmap_file_path, "wb") as f:
        f.write(base64.b64decode(heatmap_b64))

    return {
        "summary_statistics": stats,
        "missing_values": missing,
        "heatmap_file_path": heatmap_file_path,
    }

EDAInsightTool = FunctionTool(
    func=perform_eda,
    name="PerformEDA",
    description="""
    Perform exploratory data analysis (EDA) on a dataset given as a file path.
    Returns summary statistics, missing value information, and a base64-encoded correlation heatmap.
    """
)


def generate_markdown_report(overview: str, insights: str, visuals_file_paths: str, conclusions: str) -> Dict[str, str]:
    """
    Generate a Markdown report from the provided sections.
    Where:
    visuals_file_paths: A string containing paths to visualizations (output from EDA).
    """
    report_md = f"""
        # Exploratory Data Analysis Report
        
        ## Overview
        {overview}
        
        ## Key Insights
        {insights}
        
        ## Visualizations
        {visuals_file_paths}
        
        ## Conclusions
        {conclusions}
        """
    return {"markdown_report": report_md}


MarkdownReportTool = FunctionTool(
    func=generate_markdown_report,
    name="GenerateMarkdownReport",
    description="""
    Format the final EDA report in markdown using the provided content sections.
    Where:
    visuals_file_paths: A string containing paths to visualizations (output from EDA).
    """
)


def critique_text(text: str) -> Dict[str, Any]:
    """Critique the provided text and return feedback and a summary."""
    feedback = []

    if len(text.strip()) < 100:
        feedback.append("The text is too short. Consider adding more detailed explanations.")
    if "TODO" in text:
        feedback.append("There are unfinished TODO items that need attention.")
    if text.count('.') < 3:
        feedback.append("Consider breaking down or expanding sentences for clarity.")

    summary = "Review complete. Improve detail, complete sections, and enhance clarity where needed."
    return {"feedback": feedback, "summary": summary}


TextCriticTool = FunctionTool(
    func=critique_text,
    name="TextCritique",
    description="Review and provide constructive feedback on text outputs. Returns feedback and a summary."
)


def create_team(model_client : OpenAIChatCompletionClient) -> SelectorGroupChat:
    """Create a team of agents for planning and executing tasks related to data analysis."""

    # AdminAgent with an enhanced planning role
    admin_agent = AssistantAgent(
        name="AdminAgent",
        description="Strategic planner: breaks down complex tasks and delegates to the team.",
        model_client=model_client,
        system_message="""
        You are the AdminAgent, the strategic planner of a data‐analysis team.
        Your role is to break down complex tasks into manageable steps and delegate them to the appropriate agents.
        
        An sample task flow is as follows:
        - Use DataPreparationAgent to LoadCSV using 'file_path' so you get the 'data' as a list of records and a view of the dataset.
        - Then, the DataPreparationAgent cleans the df loading from disk and using the tool CleanDataFrame with a list of 'actions' (like dropna, fillna, etc.).
        - Pass the cleaned 'data' output_file_path to the EDAAgent as input.
        - EDAAgent performs EDA using PerformEDA.
        - Using the results from EDAAgent, the report_generator_agent then makes a Markdown report passing overview: str, insights: str, visuals: str, conclusions: str, to the GenerateMarkdownReport tool.
        - The results are summarized into a markdown report by ReportGeneratorAgent.
        - CriticAgent reviews the final markdown.
        - Repeat the process until the report is finalized. ONLY IF NECESSARY, do not overdue the process even though the CriticAgent asserts is needed. Be efficient.
        - Use ExecutorAgent to execute any code that needs validation.
        - End with TERMINATE.
        
        Team members:
         - DataPreparationAgent: cleans and preprocesses data from file_path.
         - EDAAgent: performs statistical summarization and visualizations from file_path.
         - ReportGeneratorAgent: assembles the final report.
         - CriticAgent: reviews for clarity/actionability.
         - ExecutorAgent: validates code/results.
         
         DO NOT USE the word 'TERMINATE' in any of your responses or your initial planning or internal reasoning.
         Using the word 'TERMINATE' will cause the chat to end. This is reserved solely for accepting the final output of the ReportGeneratorAgent.
         
         Do not ask for user input or clarification or if to execute next steps. Assume all necessary information is provided in the task description and you are responsible for planning and delegating tasks to the appropriate agents.
    """,
    )

    data_preparation_tools = [PandasDataPreparationTool]  # LoadCSVTool can be added here too
    eda_tools = [EDAInsightTool]
    report_generator_tools = [MarkdownReportTool]
    critic_tools = [TextCriticTool]

    data_preparation_agent = AssistantAgent(
        name="DataPreparationAgent",
        description="Cleans and preprocesses data to prepare it for analysis.",
        model_client=model_client,
        system_message="""
    You are the DataPreparationAgent, responsible for preparing raw data for analysis.

    Your tasks include:
     - Handling missing values (e.g., imputation or removal).
     - Encoding categorical variables.
     - Normalizing or scaling features if necessary.
     - Removing duplicates and correcting data inconsistencies.
     - Providing a clean, well-structured dataset to the EDAAgent.

    You do not analyze or visualize the data—that is the role of the EDAAgent.

    Document any changes you make to the data and provide a summary of the cleaned dataset format (e.g., column types, shapes).
    Use markdown formatting where appropriate for clarity.
    """,
        tools=data_preparation_tools,
    )

    EDA_agent = AssistantAgent(
        name="EDAAgent",
        description="Performs statistical analysis, visualizes data, and extracts key insights.",
        model_client=model_client,
        system_message="""
        You are the EDAAgent, responsible for performing exploratory data analysis (EDA).
        Your tasks include:
         - Analyzing the structure and distribution of data.
         - Calculating summary statistics (mean, median, missing values, etc.).
         - Creating clear and informative visualizations (e.g., histograms, scatter plots, correlation matrices).
         - Extracting meaningful insights and trends from the data.
        You do not clean the data—that is the job of the DataPreparationAgent.

        Present your results clearly using markdown formatting when applicable, and include visual outputs if possible.
        Keep explanations concise and focused on actionable insights.
        """,
        tools=eda_tools,
    )

    report_generator_agent = AssistantAgent(
        name="ReportGeneratorAgent",
        description="Assembles a structured and readable EDA report based on findings from other agents.",
        model_client=model_client,
        system_message="""
    You are the ReportGeneratorAgent.

    Your job is to compile a professional, well-structured report based on:
     - Preprocessed data from the DataPreparationAgent.
     - Analytical findings and visualizations from the EDAAgent.
     - Feedback from the CriticAgent, if available.

    Your report must include:
     - A title and brief introduction.
     - An overview of the dataset.
     - Key findings with summaries and visuals.
     - Conclusions and suggested next steps.

    Use clear markdown formatting for sections, bullet points, and embedded images if present.
    Avoid raw code — your output should be clean, concise, and suitable for non-technical stakeholders.
    """,
        tools=report_generator_tools,
    )

    critic_agent = AssistantAgent(
        name="CriticAgent",
        description="Reviews outputs and provides constructive feedback to improve clarity, accuracy, and impact, without overhauling/excessive .",
        model_client=model_client,
        system_message="""
    You are the CriticAgent.

    Your job is to review the outputs (especially from the EDAAgent and ReportGeneratorAgent) and:
     - Identify unclear language, errors, or missing explanations.
     - Suggest improvements to clarity, insightfulness, or presentation.
     - Ensure conclusions are well-supported by the data.

    Be specific and constructive in your feedback.
    Summarize the strengths of the output before offering revisions.
    Use markdown for clear formatting and make your feedback easy to act on.
    Do not redo the original work — just critique it.

    End with a brief "Critic Summary" paragraph summarizing your suggestions.
    """,
        tools=critic_tools,
    )

    executor_agent = CodeExecutorAgent(
        name="ExecutorAgent",
        code_executor=LocalCommandLineCodeExecutor(work_dir="eda_working_dir"),
        description="Validates and executes code received from other agents, returns results or errors."
    )

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    participants = [
        admin_agent,
        data_preparation_agent,
        EDA_agent,
        report_generator_agent,
        critic_agent,
        executor_agent,
    ]

    # AdminAgent handles planning first
    def candidate_func(messages):
        if messages and messages[-1].source == "user":
            return ["AdminAgent"]
        # Always return at least one candidate
        return [agent.name for agent in participants]  # or a fallback like ["AdminAgent"]

    selector_prompt = """Select the next agent to speak based on roles and conversation.

    {roles}

    Conversation history:
    {history}

    Choose one agent from {participants} to continue the task. Ensure planning tasks come from AdminAgent first, then execution proceeds according to role responsibilities.
    """

    team = SelectorGroupChat(
        participants=participants,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,
        candidate_func=candidate_func,
    )
    return team


async def main() -> None:

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    team = create_team(model_client)
    df_sample = sns.load_dataset("titanic")

    # get the first 20 records only
    df_head = df_sample.head(2)

    # pretty print in terminal
    print("Sample of the Titanic dataset (first 2 records):")
    print(json.dumps(df_head.to_dict(orient='records'), indent=2))
    print("--" * 80)

    task = (
        "Perform a full EDA process on the 'titanic.csv' dataset. "
        "Use 'titanic.csv' as the `file_path`"
    )
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())