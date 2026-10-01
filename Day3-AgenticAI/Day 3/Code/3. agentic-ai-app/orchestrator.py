import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

# -----------------------------------
# Setup
# -----------------------------------

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = "gpt-5-nano"
STATE_FILE = "saga_state.json"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# State Store
# -----------------------------------

def save_state(state):

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def load_state():

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)

    return {
        "status": "NOT_STARTED",
        "completed_agents": [],
        "metadata": {}
    }

# -----------------------------------
# Utility Functions
# -----------------------------------

def write_output(filename, content):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created: {path}")

    return path


def delete_output(filename):

    path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")


def remove_agent_state(agent_name, metadata_keys, state):

    if agent_name in state["completed_agents"]:
        state["completed_agents"].remove(agent_name)

    for key in metadata_keys:
        state["metadata"].pop(key, None)

    save_state(state)

# -----------------------------------
# PDF Reader
# -----------------------------------

def read_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

# -----------------------------------
# OpenAI Helper
# -----------------------------------

def ask_llm(system_prompt, user_prompt):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content

# -----------------------------------
# Agents
# -----------------------------------

class ValidationAgent:

    name = "ValidationAgent"

    def execute(self, paper_text):

        print("\n[Validation Agent]")

        result = ask_llm(
            "You validate research papers. "
            "Briefly decide whether the paper looks academically valid. "
            "Start the answer with VALID or INVALID.",
            paper_text[:6000]
        )

        print(result)

        if result.upper().startswith("INVALID"):
            raise Exception("Validation failed")

        write_output("validation.txt", result)

        return {
            "validation": result
        }

    def compensate(self, state):

        print("Compensating Validation Agent")

        delete_output("validation.txt")

        remove_agent_state(
            self.name,
            ["validation"],
            state
        )

# -----------------------------------

class ReviewAgent:

    name = "ReviewAgent"

    def execute(self, paper_text):

        print("\n[Review Agent]")

        review = ask_llm(
            "Act as an academic reviewer. "
            "Provide concise review comments with:\n"
            "1. Strengths\n"
            "2. Weaknesses\n"
            "3. Suggestions",
            paper_text[:6000]
        )

        write_output("review.txt", review)

        return {
            "review_comments": review
        }

    def compensate(self, state):

        print("Compensating Review Agent")

        delete_output("review.txt")

        remove_agent_state(
            self.name,
            ["review_comments"],
            state
        )

# -----------------------------------

class SummaryAgent:

    name = "SummaryAgent"

    def execute(self, paper_text):

        print("\n[Summary Agent]")

        # Simulate failure for testing rollback
        if "FAIL_SUMMARY" in paper_text:
            raise Exception("Simulated Summary Failure")

        summary = ask_llm(
            "Summarize this research paper in approximately 300 words.",
            paper_text[:6000]
        )

        write_output("summary.txt", summary)

        return {
            "summary": summary
        }

    def compensate(self, state):

        print("Compensating Summary Agent")

        delete_output("summary.txt")

        remove_agent_state(
            self.name,
            ["summary"],
            state
        )

# -----------------------------------
# Saga Orchestrator
# -----------------------------------

class SagaOrchestrator:

    def __init__(self):

        self.completed = []

        self.state = {
            "status": "RUNNING",
            "completed_agents": [],
            "metadata": {}
        }

        save_state(self.state)

    def update_state(self, agent_name, metadata):

        self.state["completed_agents"].append(agent_name)

        self.state["metadata"].update(metadata)

        save_state(self.state)

        print(f"\nState Updated After {agent_name}")

    def rollback(self):

        print("\n================================")
        print("STARTING COMPENSATION")
        print("================================")

        for agent in reversed(self.completed):

            agent.compensate(self.state)

        self.state["status"] = "ROLLED_BACK"

        save_state(self.state)

        print("\nFinal State:")

        print(json.dumps(self.state, indent=4))

        print("\n================================")
        print("ROLLBACK COMPLETE")
        print("================================")

    def run(self, pdf_path):

        paper_text = read_pdf(pdf_path)

        validation = ValidationAgent()
        review = ReviewAgent()
        summary = SummaryAgent()

        try:

            # -------------------------
            # Validation
            # -------------------------

            v = validation.execute(paper_text)

            self.completed.append(validation)

            self.update_state(
                validation.name,
                v
            )

            # -------------------------
            # Review
            # -------------------------

            r = review.execute(paper_text)

            self.completed.append(review)

            self.update_state(
                review.name,
                r
            )

            # -------------------------
            # Summary
            # -------------------------

            s = summary.execute(paper_text)

            self.completed.append(summary)

            self.update_state(
                summary.name,
                s
            )

            self.state["status"] = "COMPLETED"

            save_state(self.state)

            print("\n================================")
            print("SAGA SUCCESS")
            print("================================")

            print("\nValidation")
            print(v["validation"])

            print("\nReview")
            print(r["review_comments"])

            print("\nSummary")
            print(s["summary"])

        except Exception as e:

            print("\nFailure:", e)

            self.state["status"] = "FAILED"

            save_state(self.state)

            self.rollback()

# -----------------------------------
# Run
# -----------------------------------

pdf_file = "paper.pdf"

orchestrator = SagaOrchestrator()

orchestrator.run(pdf_file)