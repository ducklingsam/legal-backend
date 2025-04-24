from sentence_transformers import SentenceTransformer, util
from weasyprint import HTML
import difflib
import torch
from rospatent_connect import *
import time


# class LLaMa:
#     def __init__(self, key=Config.LLAMA_KEY):
#         self._url = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#         self.__key = key

#     def llm(self, query):
#         parameters = {
#             "max_tokens": 100,
#             "temperature": 0.01,
#             "top_k": 50,
#             "top_p": 0.95,
#             "return_full_text": False
#         }
        
#         prompt = (
#             "Пожалуйста, возвращайте только ключевую фразу без префиксов или служебных тегов. "
#             "Вы являетесь научным помощником, специализирующимся на поиске патентов. "
#             "Перепишите следующий исходный поисковый запрос так, чтобы он содержал только ключевую фразу, был оптимизирован для поиска патентных документов, "
#             "Но при этом сохранял ключевую фразу исходного запроса. Если запрос касается таких продуктов, как жвачка, или изменяющихся вкусов, "
#             "обязательно сохраните соответствующую ключевую фразу. Возвращайте только ключевую фразу без дополнительных комментариев. "
#             "Вот запрос: {query}"
#         )
        
#         headers = {
#             'Authorization': f'Bearer {self.__key}',
#             'Content-Type': 'application/json'
#         }
        
#         prompt = prompt.replace("{query}", query)
#         # print(prompt)
        
#         payload = {
#             "inputs": prompt,
#             "parameters": parameters
#         }
        
#         response = requests.post(self._url, headers=headers, json=payload)
#         # print(response.json())
#         response_text = response.json()[0]['generated_text'].strip()
#         print(response_text.split("\n")[-1])
#         return response_text.split("\n")[-1]

class SBERT:
    def __init__(self):
        self._model = SentenceTransformer('all-MiniLM-L6-v2')

    def find_simmilar(self, input: str, texts: list):
        input_embedding = self._model.encode(input, convert_to_tensor=True)
        text_embeddings = self._model.encode(texts, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(input_embedding, text_embeddings)
        top_results = torch.topk(cosine_scores, k=1)
        print(texts[top_results[1][0]])
        self.create_report(input, texts[top_results[1][0]], top_results[0][0].item())

        # for score, idx in zip(top_results[0][0], top_results[1][0])):
            # print(f"Text: {texts[idx]} \nScore: {score.item():.4f}\n")

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
        # matcher = difflib.SequenceMatcher(None, input_text, most_similar_text)
        # matches = [match for match in matcher.get_matching_blocks() if match.size > 0]

        highlighted_input, highlighted_similar = self.highlight_by_tokens(input_text, most_similar_text)
        # highlighted_similar = self.highlight_by_tokens(most_similar_text, matches)

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

        # Convert the HTML to a PDF.
        output_pdf = "highlighted_texts.pdf"
        HTML(string=html_content).write_pdf(output_pdf)
        print(f"PDF generated: {output_pdf}")



if __name__ == "__main__":
    # llm = LLaMa()
    # title = llm.llm(input_text)
    # print(title)
    input_text = "Недостатком ракеты является расположение перьев стабилизатора, до ее старта, внутри и снаружи корпуса с продольными пазами, который находится в задней стороне от двигателя, а перья входят в пазы, соединяются между собой внутри корпуса и в их задней части, выполнены подвижными в радиальном направлении. В движении ракеты перья стабилизатора давлением газа из сопла выжимаются наружу, в бок от корпуса. Поэтому в полете ракета имеет выдвинутый в стороны стабилизатор. Кроме того, сопло такой ракеты выполнено дозвуковым. Это увеличивает габаритные размеры ракеты и сопротивление ее движения, не позволяет улучшить стабилизацию и точность полета, особенно в момент начала движения и разгона ракеты."
    start_api = time.perf_counter()
    patents = PatentSearch()
    patents.search_by_natural(query={"qn": input_text, "pre_tag": "", "post_tag": "", "limit": 100})
    texts = patents.get_parsed_data()
    # print(len(texts))
    end_api = time.perf_counter()
    print(f"[API Роспатента] Время выполнения: {end_api - start_api:.4f} секунд")
    print(len(texts))
    start_sbert = time.perf_counter()
    test = SBERT()
    test.find_simmilar(input_text, texts)
    end_sbert = time.perf_counter()
    print(f"[SBERT Сравнение] Время выполнения: {end_sbert - start_sbert:.4f} секунд")
    # input_texts = [
    #     "Недостатком ракеты является расположение перьев стабилизатора, до ее старта, внутри и снаружи корпуса с продольными пазами, который находится в задней стороне от двигателя, а перья входят в пазы, соединяются между собой внутри корпуса и в их задней части, выполнены подвижными в радиальном направлении. В движении ракеты перья стабилизатора давлением газа из сопла выжимаются наружу, в бок от корпуса. Поэтому в полете ракета имеет выдвинутый в стороны стабилизатор. Кроме того, сопло такой ракеты выполнено дозвуковым. Это увеличивает габаритные размеры ракеты и сопротивление ее движения, не позволяет улучшить стабилизацию и точность полета, особенно в момент начала движения и разгона ракеты.",
    #     "1. Крыло, включающее собственно крыло, и закрылок или другое средство механизации задней кромки крыла, и предкрылок, отличающееся тем, что на задней кромке предкрылка выполнено щелевое сопло для истечения воздуха.2. Крыло по п.1, отличающееся тем, что для закрывания в нерабочем состоянии щелевое сопло снабжено створкой, являющейся одной из поверхностей предкрылка, примыкающей к его задней кромке и установленной на нем с помощью осевого шарнира, с осью, направленной вдоль размаха предкрылка.",
    # ]

    # total_api_time = 0
    # total_sbert_time = 0
    # total_patents = 0

    # for idx, input_text in enumerate(input_texts, 1):
    #     print(f"\n[{idx}] Обработка текста...")

    #     # Засекаем время API
    #     start_api = time.perf_counter()
        
    #     patents = PatentSearch()
    #     patents.search_by_natural(query={"qn": input_text, "pre_tag": "", "post_tag": "", "limit": 100})
    #     texts = patents.get_parsed_data()
        
    #     api_duration = time.perf_counter() - start_api
    #     total_api_time += api_duration
    #     total_patents += len(texts)

    #     print(f"  API время: {api_duration:.4f} сек | найдено текстов: {len(texts)}")

    #     # Засекаем время SBERT
    #     start_sbert = time.perf_counter()

    #     test = SBERT()
    #     test.find_simmilar(input_text, texts)

    #     sbert_duration = time.perf_counter() - start_sbert
    #     total_sbert_time += sbert_duration

    #     print(f"  SBERT время: {sbert_duration:.4f} сек")

    # # Результаты
    # n = len(input_texts)
    # avg_api_time = total_api_time / n
    # avg_sbert_time = total_sbert_time / n
    # avg_texts_found = total_patents / n

    # print("\n--- СРЕДНИЕ ВРЕМЕННЫЕ ПОКАЗАТЕЛИ ---")
    # print(f"Среднее время API: {avg_api_time:.4f} секунд")
    # print(f"Среднее время SBERT: {avg_sbert_time:.4f} секунд")
    # print(f"Среднее количество найденных текстов: {avg_texts_found:.1f}")