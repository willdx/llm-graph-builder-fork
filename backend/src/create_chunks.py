from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph
import logging
import os
from src.document_sources.youtube import get_chunks_with_timestamps

logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


class CreateChunksofDocument:
    def __init__(self, pages: list[Document], graph: Neo4jGraph):
        self.pages = pages
        self.graph = graph

    def split_file_into_chunks(self):
        """
        Split a list of documents(file pages) into chunks of fixed size.

        Args:
            pages: A list of pages to split. Each page is a list of text strings.

        Returns:
            A list of chunks each of which is a langchain Document.
        """
        logging.info("Split file into smaller chunks")
        # number_of_chunks_allowed = int(os.environ.get('NUMBER_OF_CHUNKS_ALLOWED'))
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
        if "page" in self.pages[0].metadata:
            chunks = []
            for i, document in enumerate(self.pages):
                page_number = i + 1
                for chunk in text_splitter.split_documents([document]):
                    chunks.append(
                        Document(
                            page_content=chunk.page_content,
                            metadata={"page_number": page_number},
                        )
                    )

        elif "length" in self.pages[0].metadata:
            chunks_without_timestamps = text_splitter.split_documents(self.pages)
            chunks = get_chunks_with_timestamps(
                chunks_without_timestamps, self.pages[0].metadata["source"]
            )
        else:
            chunks = text_splitter.split_documents(self.pages)
        return chunks

    def split_content_into_chunks(self, content: str):
        """
        将单个内容拆分成多个chunks

        Args:
            content: 要拆分的内容字符串

        Returns:
            一个包含多个Document对象的列表,每个对象代表一个chunk
        """
        logging.info("将内容拆分成更小的chunks")
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = text_splitter.split_text(content)
        return [Document(page_content=chunk) for chunk in chunks]
