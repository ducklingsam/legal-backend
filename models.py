from sentence_transformers import SentenceTransformer, util
from weasyprint import HTML
import torch
from rospatent_connect import *


class SBERT:
    def __init__(self):
        self._model = SentenceTransformer("./fine_tuned_sbert_patents")

    def find_simmilar(self, input: str, texts: list):
        input_embedding = self._model.encode(input, convert_to_tensor=True)
        text_embeddings = self._model.encode(texts, convert_to_tensor=True)

        cosine_scores = util.cos_sim(input_embedding, text_embeddings)
        top_results = torch.topk(cosine_scores, k=1)
        print(texts[top_results[1][0]])
        self.create_report(
            input, texts[top_results[1][0]], top_results[0][0].item())

    def highlight_by_tokens(self, input_text, similar_text):
        tokens_input = input_text.split()
        tokens_similar = similar_text.split()

        lower_tokens_input = [token.lower() for token in tokens_input]
        lower_tokens_similar = [token.lower() for token in tokens_similar]

        common_tokens = set(lower_tokens_input) & set(lower_tokens_similar)

        def highlight(text):
            return " ".join(
                [f"<span class='highlight'>{word}</span>" if word.lower() in common_tokens else word
                 for word in text.split()]
            )

        return highlight(input_text), highlight(similar_text)

    def create_report(self, input_text: str, most_similar_text: str, score):
        highlighted_input, highlighted_similar = self.highlight_by_tokens(
            input_text, most_similar_text)

        html_content = f"""
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.5;
            }}
            .highlight {{
            background-color: yellow;
            }}
            .section {{
            margin-bottom: 30px;
            }}
            h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            }}
        </style>
        </head>
        <body>
        <div class="section">
            <h2>Input Text</h2>
            <p>{highlighted_input}</p>
        </div>
        <div class="section">
            <h2>Most Similar Text</h2>
            <p>{highlighted_similar}</p>
        </div>
        <div class="section">
            <p><b>Similarity score:</b> {score:.4f}</p>
        </body>
        </html>
        """

        output_pdf = "highlighted_texts.pdf"
        HTML(string=html_content).write_pdf(output_pdf)
        print(f"PDF generated: {output_pdf}")


if __name__ == "__main__":
    input_text = "Недостатком ракеты является расположение перьев стабилизатора, до ее старта, внутри и снаружи корпуса с продольными пазами, который находится в задней стороне от двигателя, а перья входят в пазы, соединяются между собой внутри корпуса и в их задней части, выполнены подвижными в радиальном направлении. В движении ракеты перья стабилизатора давлением газа из сопла выжимаются наружу, в бок от корпуса. Поэтому в полете ракета имеет выдвинутый в стороны стабилизатор. Кроме того, сопло такой ракеты выполнено дозвуковым. Это увеличивает габаритные размеры ракеты и сопротивление ее движения, не позволяет улучшить стабилизацию и точность полета, особенно в момент начала движения и разгона ракеты."
    patents = PatentSearch()
    patents.search_by_natural(
        query={"qn": input_text, "pre_tag": "", "post_tag": "", "limit": 1})
    texts = patents.get_parsed_data()
    test = SBERT()
    test.find_simmilar(input_text, texts)
