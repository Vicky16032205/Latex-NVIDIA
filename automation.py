import asyncio
import os
import nest_asyncio
import sys
import re

from llm_prompt import LLMPrompt
from parser import LlamaPDFParser
from retrieval import MilvusEmbeddingManager
from ToLatex import md_to_latex
from usegemini import ModelGemini



nest_asyncio.apply()


class PDFToMilvusAutomation:
    def __init__(self, pdf_paths=None, output_dir=None):
        self.pdf_paths = pdf_paths or []
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.manager = MilvusEmbeddingManager()

    def remove_initial_numbers(self, text):
        return re.sub(r'^\s*[\d\.]+\s*', '', text)

    def process_pdfs_and_dump_to_milvus(self):
        """
        Converts each PDF to Markdown, then to JSON, and inserts the JSON into Milvus.
        """
        if not self.output_dir:
            raise ValueError("Output directory is required for PDF processing.")

        for pdf_path in self.pdf_paths:
            print(f"Processing: {pdf_path}")
            try:
                # Define output paths for Markdown and JSON
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                md_path = os.path.join(self.output_dir, f"{base_name}.md")
                json_path = os.path.join(self.output_dir, f"{base_name}.json")
                image_path = os.path.join(self.output_dir, base_name)

                # Parse the PDF and generate JSON
                parser = LlamaPDFParser(pdf_path, md_path, json_path, image_path)
                parser.convert_md_to_json()  # This converts the PDF to Markdown, then to JSON

                # Insert JSON into Milvus
                print(f"Inserting JSON into Milvus for {base_name}")
                self.manager.process_and_insert_json(json_path)
                self.manager.create_indexes(base_name)

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    def perform_vector_search(self, query=None, anns_field="sub_heading_embedding", limit=5, threshold=0.80):
        """
        Performs a vector search on the data in Milvus.
        If no query is provided, performs default searches.
        """
        text_results = []

        if query:
            print(f"Performing content-based search for query: {query}")
            text_results = self.manager.query(query, anns_field=anns_field, limit=limit, threshold=threshold)
            print(f"Performing Image content search for query: {query}")
            content_results = self.manager.query(query, anns_field="content_embedding", limit=1, threshold=0.75)
        
        print("Performing default searches...")
        default_results = self.manager.perform_default_queries()

        return {
            'query': query,
            'user_based_search': text_results,
            'default_results': default_results,
            'content_results': content_results
        }
    

    async def generate_responses(self, search_result):
        """
        Generate responses for all sections concurrently using asyncio.
        """
        get_prompt = LLMPrompt()
        response_gemini = ModelGemini()

        # Create all prompts
        prompts = {
            "user_based": get_prompt.prompt_for_user_based_search(search_result),
            "abstract": get_prompt.prompt_for_abstract(search_result),
            "intro": get_prompt.prompt_for_intro(search_result),
            "methodology": get_prompt.prompt_for_methodology(search_result),
            "result": get_prompt.prompt_for_result(search_result),
            "conclusion": get_prompt.prompt_for_conclusion(search_result),
            "reference": get_prompt.prompt_for_reference(search_result),
        }

        # Run all LLM calls asynchronously
        responses = await asyncio.gather(*[
            response_gemini.gemini_response(prompt) for prompt in prompts.values()
        ])

        # Map responses back to section names
        response_data = dict(zip(prompts.keys(), responses))

        lit_review = await response_gemini.gemini_response(get_prompt.prompt_for_lit_review(response_data['reference']))

        image_path = "images\\Project_Workflow.png"
        caption_prompt = None

        for field, collection in search_result.get("content_results", {}).items():
            if isinstance(collection, list) and collection:
                for item in collection:
                    if "image" in item and item["image"] and item["image"] != "No image available":
                        image_path = item["image"]  # Pick the first valid image path
                        caption_prompt = item.get("text", None)  # Pick the text from the same field
                        break 

        caption = ""
        if caption_prompt:
            caption = await response_gemini.gemini_response(get_prompt.prompt_for_caption(caption_prompt))


        # Write to Markdown file
        with open('./paper.md', 'w', encoding='utf-8') as data:
            data.write("# Review Paper\n\n")
            data.write(f"## Abstract\n{response_data['abstract']}\n\n")
            data.write(f"## Introduction\n{response_data['intro']}\n\n")
            data.write(f"## Litrature Review\n{lit_review}\n\n")
            data.write(f"## Methodology\n{response_data['methodology']}\n\n")
            data.write(f"{response_data['user_based']}\n\n")
            if image_path != "No image available":
                data.write(f"![Figure]({image_path})\n\n")
                if caption:
                    data.write(f"**Figure Caption:** {caption}\n\n")
            data.write(f"## Results\n{response_data['result']}\n\n")
            data.write(f"## Conclusion\n{response_data['conclusion']}\n\n")
            data.write(f"## References\n{response_data['reference']}\n\n")

async def main():
    # Get mode, list of PDF files, and optional output directory or query
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Dumping to Milvus: python automation.py dump <pdf1> <pdf2> ... <output_directory>")
        print("  Search: python automation.py search [<query>]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "dump":
        if len(sys.argv) < 4:
            print("Usage: python automation.py dump <pdf1> <pdf2> ... <output_directory>")
            sys.exit(1)

        pdf_files = sys.argv[2:-1]
        output_directory = sys.argv[-1]

        # Initialize the automation process for dumping
        automation = PDFToMilvusAutomation(pdf_files, output_directory)

        # Process PDFs to JSON and insert into Milvus
        automation.process_pdfs_and_dump_to_milvus()

    elif mode == "search":
        user_query = sys.argv[2] if len(sys.argv) > 2 else None

        # Initialize the automation process for search
        automation = PDFToMilvusAutomation()

        # Perform vector searches
        search_result = automation.perform_vector_search(query=user_query)

        os.makedirs("./extracted", exist_ok=True)

        with open('./extracted/search_result.txt','w',encoding='utf-8') as data:
            data.write(str(search_result))
        
        # Generate responses concurrently using asyncio
        await automation.generate_responses(search_result)

        md_to_latex("paper.md", "latex-output/output.tex", "latex-output/output.pdf")

    else:
        print("Invalid mode. Use 'dump' for dumping to Milvus or 'search' for searching.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())