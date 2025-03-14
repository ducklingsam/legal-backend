import requests
from config import *


class PatentSearch:
    """
    Класс для поиска патентов через API Роспатента.

    Атрибуты:
        __key (str): API-ключ для аутентификации.
        __data (dict | None): Хранит данные ответа API после поиска.
    """
    def __init__(self, api_key=Config.API_KEY):
        """
        Инициализирует объект поиска патентов.

        Параметры:
            api_key (str): API-ключ (по умолчанию берётся из Config.API_KEY).
            __data (dict | None): Хранит данные ответа API после поиска.
            __parsed_data (dict): Хранит данные после обработки.
        """
        self.__key = api_key
        self.__data = None
        self.__parsed_data = {}

    def parse_json(self,):
        """
        Обрабатывает JSON-ответ API и выводит информацию о найденных патентах.

        Метод извлекает заголовок и описание из каждого найденного патента
        и выводит их в консоль. Если данные отсутствуют, ничего не выводится.
        """
        for hit in self.__data.get("hits", {}):
            title = hit.get("snippet", {}).get("title", None)
            desc = hit.get("snippet", {}).get("description", None)  
            self.__parsed_data[title] = desc
    
    def search_by_natural(self, query=Config.EXAMPLE_QUERY, headers=Config.EXAMPLE_HEADERS, url=Config.URL_SEARCH):
        """
        Выполняет поиск патентов по естественному языковому запросу.

        Параметры:
            query (dict): JSON-запрос (по умолчанию берётся из Config.EXAMPLE_QUERY).
            headers (dict): HTTP-заголовки (по умолчанию из Config.EXAMPLE_HEADERS).
            url (str): URL API для поиска (по умолчанию из Config.URL_SEARCH).

        Метод отправляет POST-запрос к API, получает ответ и передаёт данные
        в метод json_parse(). В случае ошибки выводит сообщение с кодом ответа.
        """
        response = requests.post(url, json=query, headers=headers)
        
        if response.status_code == 200:
            self.__data = response.json()
            self.parse_json()  
        else:
            print(f"Ошибка {response.status_code}: {response.text}")

    def get_parsed_data(self,):
        """
        Выводит обработанные данные.
        """
        arr = []

        for k, v in self.__parsed_data.items():
            arr.append(v)

        return arr


if __name__ == "__main__":
    test = PatentSearch()
    test.search_by_natural()
    test.get_parsed_data()