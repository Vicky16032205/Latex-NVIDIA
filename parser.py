import fitz
import json
import os
import nest_asyncio
import re

from dotenv import load_dotenv
from llama_parse import LlamaParse
# from llama_index.embeddings.nvidia import NVIDIAEmbedding
from sentence_transformers import SentenceTransformer


nest_asyncio.apply()


class LlamaPDFParser:
    def __init__(self, pdf_path, output_md_path, output_json_path, image_output_folder):
        load_dotenv()
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        # self.nim_api_key = os.getenv("NIM_API_KEY")
        if not self.api_key:
            raise ValueError("API key for Llama Cloud is not set in the .env file.")
        os.environ["LLAMA_CLOUD_API_KEY"] = self.api_key
        # os.environ["NIM_API_KEY"] = self.nim_api_key

        # self.embedding_model = NVIDIAEmbedding(
        #     model="nvidia/nv-embedqa-e5-v5",
        #     truncate="END",
        #     api_key=self.nim_api_key
        # )
        self.embedding_model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

        self.pdf_path = pdf_path
        self.output_md_path = output_md_path
        self.output_json_path = output_json_path
        self.image_output_path = image_output_folder
        self.documents, self.images_with_caption = self._parse_pdf_to_markdown()

    def _clean_heading(self, heading):
        """Helper function to clean and normalize headings."""
        return heading.strip("# ").strip()

    def _parse_pdf_to_markdown(self):
        """
        Parses the input PDF file using LlamaParse and saves it as a Markdown file.
        """
        parser = LlamaParse(
            result_type="markdown",
            premium_mode=True,
        )
        try:
            # Load data from the PDF (returns a list of Document objects)
            documents = parser.load_data(self.pdf_path)
            if not documents:
                raise ValueError("No data was parsed from the provided PDF.")

            # Extract text from each document object and join them into a single string
            markdown_content = "\n\n".join([doc.text for doc in documents])  # Assuming each 'doc' has a 'text' attribute
            
            # Save Markdown content to a file
            os.makedirs(os.path.dirname(self.output_md_path), exist_ok=True)
            with open(self.output_md_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)

            images_with_caption = self._extract_images_with_captions()

            # Append images to Markdown file
            with open(self.output_md_path, "a", encoding="utf-8") as md_file:
                for img in images_with_caption:
                    md_file.write(f"\n![Image]({img['metadata']['image']})\n")
                    md_file.write(f"\n**Caption:** {img['metadata']['caption']}\n\n")            
            
            return markdown_content, images_with_caption  # Return the Markdown content as a string
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {e}")
        

    def parse_all_images(self, filename, page, pagenum, text_blocks):
        """Extract images from a PDF page."""
        image_docs = []
        image_info_list = page.get_image_info(xrefs=True)
        page_rect = page.rect

        for image_info in image_info_list:
            xref = image_info['xref']
            if xref == 0:
                continue

            img_bbox = fitz.Rect(image_info['bbox'])
            if img_bbox.width < page_rect.width / 20 or img_bbox.height < page_rect.height / 20:
                continue

            extracted_image = page.parent.extract_image(xref)
            image_data = extracted_image["image"]
            imgrefpath = os.path.join(os.getcwd(), f"{filename}")
            os.makedirs(imgrefpath, exist_ok=True)
            image_path = os.path.join(imgrefpath, f"image{xref}-page{pagenum}.png")
            with open(image_path, "wb") as img_file:
                img_file.write(image_data)
            before_text, after_text = self.extract_text_around_item(text_blocks, img_bbox, page.rect.height)
            if before_text == "" and after_text == "":
                continue

            image_description = " "

            caption = before_text.replace("\n", " ") + image_description + after_text.replace("\n", " ")

            image_metadata = {
                "source": f"{filename}-page{pagenum}-image{xref}",
                "image": image_path,
                "caption": caption,
                "type": "image",
                "page_num": pagenum
            }
            image_docs.append({
                "text": "This is an image with the caption: " + caption,
                "metadata": image_metadata
            })
            
            # breakpoint()

        return image_docs


    def _extract_images_with_captions(self):
        """
        Extracts images from the PDF with captions and saves them to a folder.
        Returns a list of image documents with metadata.
        """
        os.makedirs(self.image_output_path, exist_ok=True)
        doc = fitz.open(self.pdf_path)
        image_docs = []

        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("blocks")
            image_docs.extend(self.parse_all_images(self.image_output_path, page, page_num + 1, text_blocks))

        return image_docs
    
    
    
    def extract_text_around_item(self, text_blocks, bbox, page_height, threshold_percentage=0.1):
        """Extract text above and below a given bounding box on a page."""
        before_text, after_text = "", ""
        vertical_threshold_distance = page_height * threshold_percentage
        horizontal_threshold_distance = bbox.width * threshold_percentage

        for block in text_blocks:
            block_bbox = fitz.Rect(block[:4])
            vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
            horizontal_overlap = max(0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

            if vertical_distance <= vertical_threshold_distance and horizontal_overlap >= -horizontal_threshold_distance:
                if block_bbox.y1 < bbox.y0 and not before_text:
                    before_text = block[4]
                elif block_bbox.y0 > bbox.y1 and not after_text:
                    after_text = block[4]
                    break

        return before_text, after_text



    def _parse_markdown_to_json(self, md_content):
        """Parse Markdown file into a hierarchical JSON format."""
        main_title = ""
        hierarchy = {}
        current_levels = []

        for line in md_content.splitlines():
            # Match for main title (only the first # heading)
            if re.match(r"^# ", line):
                if not main_title:  # Capture the first main title
                    main_title = self._clean_heading(line)
                continue

            # Match for section and subheadings
            heading_match = re.match(r"^(#+) (.+)", line)
            if heading_match:
                level = len(heading_match.group(1))  # Determine heading level
                heading_text = heading_match.group(2)

                # Adjust the current level hierarchy
                while len(current_levels) >= level:
                    current_levels.pop()
                current_levels.append(heading_text)

                # Build metadata for the current section
                metadata = {
                    "main title": main_title,
                    "section title": current_levels[0] if len(current_levels) > 0 else "",
                    "sub heading": current_levels[1] if len(current_levels) > 1 else "",
                }

                # Navigate to the appropriate level in the hierarchy
                current_level = hierarchy
                for lvl in current_levels[:-1]:
                    current_level = current_level.setdefault(lvl, {"content": "", "subheadings": {}})["subheadings"]

                # Create a new entry for the current heading
                if current_levels[-1] not in current_level:
                    current_level[current_levels[-1]] = {"content": "", "metadata": metadata, "subheadings": {}}

            else:
                # Add content to the most recent heading
                if current_levels:
                    current_level = hierarchy
                    for lvl in current_levels[:-1]:
                        current_level = current_level[lvl]["subheadings"]
                    if current_levels[-1] not in current_level:
                        current_level[current_levels[-1]] = {
                            "content": "",
                            "metadata": {
                                "main title": main_title,
                                "section title": current_levels[0] if len(current_levels) > 0 else "",
                                "sub heading": current_levels[1] if len(current_levels) > 1 else ""
                            },
                            "subheadings": {}
                        }
                    current_level[current_levels[-1]]["content"] += line.strip() + "\n"

        return hierarchy

    def _format_hierarchy_to_json(self, hierarchy):
        """Recursive function to format the hierarchy into the desired JSON structure."""
        json_list = []
        for key, value in hierarchy.items():
            json_obj = {
                "content": value["content"].strip(),
                "metadata": value.get("metadata", {}),
                "embeddings-Main-Headding": "",
                "embeddings-Section-Headding": "",
                "embeddings-Sub-Headding": "",
                "subheadings": self._format_hierarchy_to_json(value["subheadings"]),
            }
            json_list.append(json_obj)
        return json_list

    def convert_md_to_json(self):
        """Convert Markdown file to JSON and save it to a file."""
        md_content = self.documents
        hierarchy = self._parse_markdown_to_json(md_content)
        formatted_json = self._format_hierarchy_to_json(hierarchy)

        # Add images to JSON
        for img in self.images_with_caption:
            formatted_json.append({
                "content": f"Image with caption: {img['metadata']['caption']}",
                "metadata": img["metadata"],
                "embeddings-Main-Headding": "",
                "embeddings-Section-Headding": "",
                "embeddings-Sub-Headding": "",
                "subheadings": []
            })

        # Save JSON to a file
        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(formatted_json, json_file, indent=4)

        print(f"Markdown and JSON files saved to {self.output_md_path} and {self.output_json_path}.")

    def split_heading_wise(self):
        """Splits the parsed Markdown document into a hierarchical structure."""
        return self._parse_markdown_to_json(self.documents)

    def save_cleaned_data(self, output_path):
        """
        Saves the cleaned and merged data to a Markdown file with proper hierarchical structure.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cleaned_data = self.split_heading_wise()

        def write_section(md_file, data, level=1):
            """Recursively writes sections and sub-sections to the Markdown file."""
            for heading, content in data.items():
                md_file.write(f"{'#' * level} {heading}\n\n")
                if isinstance(content, dict) and "content" in content:
                    md_file.write(f"{content['content']}\n\n")
                if isinstance(content, dict) and "subheadings" in content:
                    write_section(md_file, content["subheadings"], level + 1)

        with open(output_path, "w", encoding="utf-8") as md_file:
            write_section(md_file, cleaned_data)

        print(f"Parsed Markdown file saved to {output_path}.")

    def get_text_page_nodes(self):
        """Splits the document into nodes for each heading and subheading."""
        docs = self.split_heading_wise()
        nodes = []
        main_title = list(docs.keys())[0] if docs else "Untitled Document"

        def add_nodes(data, parent_metadata, level=1):
            for heading, content in data.items():
                node = {
                    "text": f"{heading}\n\n{content['content']}",
                    "metadata": {
                        "main title": parent_metadata["main title"],
                        "section title": heading if level == 1 else parent_metadata["section title"],
                        "sub heading": heading if level > 1 else ""
                    },
                    "embeddings-Main-Headding": "",
                    "embeddings-Section-Headding": "",
                    "embeddings-Sub-Headding": "",
                }
                nodes.append(node)
                if "subheadings" in content and content["subheadings"]:
                    add_nodes(content["subheadings"], node["metadata"], level + 1)

        add_nodes(docs, {"main title": main_title, "section title": ""})
        return nodes

    def generate_embeddings(self):
        """Generate embeddings for headings and store them in nodes."""
        embed_model = self.embedding_model
        nodes = self.get_text_page_nodes()

        for node in nodes:
            main_heading_text = node['metadata'].get('main title', '')
            section_heading_text = node['metadata'].get('section title', '')

            if main_heading_text:
                # node['embeddings-Main-Headding'] = embed_model.get_query_embedding(main_heading_text) #For nv-embed
                node['embeddings-Main-Headding'] = embed_model.encode(main_heading_text)
            if section_heading_text:
                # node['embeddings-Section-Headding'] = embed_model.get_query_embedding(section_heading_text) #For nv-embed
                node['embeddings-Section-Headding'] = embed_model.encode(section_heading_text)

        print("Embeddings generated and stored in nodes.")
        return nodes

    @staticmethod
    def custom_serializer(obj):
        """Custom serializer for unsupported objects."""
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)
    
    