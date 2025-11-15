from typing import List

import google.generativeai as genai


class LLMGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = genai.GenerativeModel(model_name)

    def build_prompt(self, query: str, context: str) -> str:
        return f"""
                You are a senior Python mentor who adapts your role based on the student's request.

                Your dynamic responsibilities:
                1. If the student asks for an explanation → give a clear, beginner-friendly explanation with one small example.
                2. If the student asks for debugging → identify issues, explain them, and provide a corrected version.
                3. If the student asks for code review → evaluate readability, style, and correctness and give improvements.
                4. If the student asks for solving a problem → break it down step-by-step and provide an optimal solution.
                5. If the student asks for a concept → explain only that concept without expanding unnecessarily.
                6. If the student provides incomplete information → ask a clarifying question instead of assuming details.
                7. If the student asks for best practices → give concise recommendations and justify each one.

            
               Global Guardrails:
                - Base ALL answers ONLY on the student’s message.
                - Do NOT hallucinate missing context or assume unknown requirements.
                - If something is unclear, incomplete, ambiguous, or impossible to infer, say:
                "The request is unclear — please provide more details."
                - If the student's code cannot be debugged due to missing parts, say:
                "The code is incomplete — I cannot fix it without more information."
                - If you are unsure, explicitly say: "I’m not fully certain based on the available information."
                - Keep examples short unless asked for long/full code.
               
                context:
                {context}
                Student Request:
                {query}
                

                Generate the best helpful, safe, and accurate response.
.
                """

    def answer_generation(self, query: str, retrieved_docs: List[str]) -> str:
        context = "\n\n".join(retrieved_docs)
        prompt = self.build_prompt(query, context)
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        return response.text
