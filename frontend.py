import gradio as gr
import requests
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8090/submit_mcq")


class MCQRequest(BaseModel):
    text: str
    question: str
    choices: List[str]


class MCQResponse(BaseModel):
    answer: str


def submit_mcq_form(
    reading_text: str,
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """
    Submit the MCQ form data to the backend server.

    Args:
        reading_text: The reading passage text
        question: The question text
        option_a: Answer option A
        option_b: Answer option B
        option_c: Answer option C
        option_d: Answer option D

    Returns:
        Response message from the backend
    """

    # Prepare the data according to MCQRequest model
    form_data = {
        "text": reading_text,
        "question": question,
        "choices": [
            "A) " + option_a,
            "B) " + option_b,
            "C) " + option_c,
            "D) " + option_d,
        ],
    }

    try:
        response = requests.post(
            BACKEND_URL,
            json=form_data,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "No answer provided")
            return f"✅ Success! Correct Answer:\n\n{answer}"
        else:
            return f"❌ Error: Server returned status code {response.status_code}. Response: {response.text}"

    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to backend server. Make sure the server is running."
    except requests.exceptions.Timeout:
        return "❌ Error: Request timed out. The server might be busy."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_mcq_interface():
    """Create the Gradio interface for the MCQ form."""

    # Sample data for demonstration
    sample_text = """This passage is adapted from George Eliot, Silas Marner. Originally published in 1861. Silas was a weaver and a notorious miser, but then the gold he had hoarded was stolen. Shortly after, Silas adopted a young child, Eppie, the daughter of an impoverished woman who had died suddenly.
    Unlike the gold which needed nothing, and must be worshipped in close-locked solitude—which was hidden away from the daylight, was deaf to the song of birds, and started to no human tones—Eppie was a creature of endless claims and ever-growing desires, seeking and loving sunshine, and living sounds, and living movements; making trial of everything, with trust in new joy, and stirring the human kindness in all eyes that looked on her. The gold had kept his thoughts in an ever-repeated circle, leading to nothing beyond itself; but Eppie was an object compacted of changes and hopes that forced his thoughts onward, and carried them far away from their old eager pacing towards the same blank limit—carried them away to the new things that would come with the coming years, when Eppie would have learned to understand how her father Silas cared for her; and made him look for images of that time in the ties and charities that bound together the families of his neighbors. The gold had asked that he should sit weaving longer and longer, deafened and blinded more and more to all things except the monotony of his loom and the repetition of his web; but Eppie called him away from his weaving, and made him think all its pauses a holiday, reawakening his senses with her fresh life, even to the old winter-flies that came crawling forth in the early spring sunshine, and warming him into joy because she had joy.
    And when the sunshine grew strong and lasting, so that the buttercups were thick in the meadows, Silas might be seen in the sunny mid-day, or in the late afternoon when the shadows were lengthening under the hedgerows, strolling out with uncovered head to carry Eppie beyond the Stone-pits to where the flowers grew, till they reached some favorite bank where he could sit down, while Eppie toddled to pluck the flowers, and make remarks to the winged things that murmured happily above the bright petals, calling “Dad-dad's” attention continually by bringing him the flowers. Then she would turn her ear to some sudden bird-note, and Silas learned to please her by making signs of hushed stillness, that they might listen for the note to come again: so that when it came, she set up her small back and laughed with gurgling triumph. Sitting on the banks in this way, Silas began to look for the once familiar herbs again; and as the leaves, with their unchanged outline and markings, lay on his palm, there was a sense of crowding remembrances from which he turned away timidly, taking refuge in Eppie’s little world, that lay lightly on his enfeebled spirit.
    As the child's mind was growing into knowledge, his mind was growing into memory: as her life unfolded, his soul, long stupefied in a cold narrow prison, was unfolding too, and trembling gradually into full consciousness.
    It was an influence which must gather force with every new year: the tones that stirred Silas’ heart grew articulate, and called for more distinct answers; shapes and sounds grew clearer for Eppie’s eyes and ears, and there was more that “Dad-dad” was imperatively required to notice and account for. Also, by the time Eppie was three years old, she developed a fine capacity for mischief, and for devising ingenious ways of being troublesome, which found much exercise, not only for Silas’ patience, but for his watchfulness and penetration. Sorely was poor Silas puzzled on such occasions by the incompatible demands of love."""

    sample_question = "Which statement best describes a technique the narrator uses to represent Silas's character before he adopted Eppie?"

    with gr.Blocks(
        title="MCQ Reading Comprehension", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# 📚 MCQ Reading Comprehension Form")
        gr.Markdown(
            "Fill out the form below with a reading passage, question, and answer choices. The backend will analyze the input and return the correct choice (currently using longest choice algorithm)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Reading text input
                reading_input = gr.Textbox(
                    label="📖 Reading Passage",
                    placeholder="Enter the reading passage here...",
                    lines=8,
                    value=sample_text,
                )

                # Question input
                question_input = gr.Textbox(
                    label="❓ Question",
                    placeholder="Enter the question here...",
                    lines=2,
                    value=sample_question,
                )
                # Answer options
                gr.Markdown("### Answer Options")
                with gr.Row():
                    option_a = gr.Textbox(
                        label="A)",
                        placeholder="Option A",
                        value="The narrator emphasizes Silas's former obsession with wealth by depicting his gold as requiring certain behaviors on his part.",
                    )
                    option_b = gr.Textbox(
                        label="B)",
                        placeholder="Option B",
                        value="The narrator underscores Silas's former greed by describing his gold as seeming to reproduce on its own.",
                    )

                with gr.Row():
                    option_c = gr.Textbox(
                        label="C)",
                        placeholder="Option C",
                        value="The narrator hints at Silas's former antisocial attitude by contrasting his present behavior toward his neighbors with his past behavior toward them.",
                    )
                    option_d = gr.Textbox(
                        label="D)",
                        placeholder="Option D",
                        value="The narrator demonstrates Silas's former lack of self-awareness by implying that he is unable to recall life before Eppie.",
                    )

                # Submit button
                submit_btn = gr.Button(
                    "📤 Submit to Backend", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                # Response display
                gr.Markdown("### 📋 Backend Response")
                response_output = gr.Textbox(
                    label="Server Response",
                    lines=10,
                    interactive=False,
                    placeholder="Response will appear here after submission...",
                )

                # Clear button
                clear_btn = gr.Button("🗑️ Clear Form", variant="secondary")

        # Event handlers
        submit_btn.click(
            fn=submit_mcq_form,
            inputs=[
                reading_input,
                question_input,
                option_a,
                option_b,
                option_c,
                option_d,
            ],
            outputs=response_output,
        )

        def clear_form():
            return ["", "", "", "", "", "", "A", ""]

        clear_btn.click(
            fn=clear_form,
            outputs=[
                reading_input,
                question_input,
                option_a,
                option_b,
                option_c,
                option_d,
                response_output,
            ],
        )

    return interface


if __name__ == "__main__":
    interface = create_mcq_interface()
    interface.launch(
        server_name=os.getenv("HOST", "localhost"),
        server_port=int(os.getenv("FRONTEND_PORT", "7860")),
        # debug=True,  # Enable debug mode
    )
