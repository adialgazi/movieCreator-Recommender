import unittest
from io import StringIO
from unittest.mock import patch
import ShowSuggesterAI
from openai import OpenAI
import pandas as pd
import numpy as np
import os

class TestTvShowSuggester(unittest.TestCase):

    client = OpenAI(api_key=os.getenv('OpenAI_API_KEY'))

    def test_excel_to_dataframe(self):
        df = ShowSuggesterAI.excel_to_dataframe('imdb_tvshows.xlsx')
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        self.assertIn('Title', df.columns)

    def test_similar_show_finding(self):
        mock_df = pd.DataFrame({'Title': ['Breaking Bad', 'Game of Thrones', 'The Wire']})
        result = ShowSuggesterAI.get_most_similar_shows(mock_df, ['Breaking Bad', 'The Wire'])
        self.assertIn('Breaking Bad', result)
        self.assertIn('The Wire', result)
        self.assertNotIn('Game of Thrones', result)

    def test_loading_embeddings_from_pickle(self):
        embeddings = ShowSuggesterAI.load_embeddings_from_pickle('tv_show_embeddings.pkl')
        self.assertIsNotNone(embeddings)
        self.assertIsInstance(embeddings, dict)

    def test_cosine_similarity(self):
        vec_a = [1, 2, 3]
        vec_b = [4, 5, 6]
        similarity = ShowSuggesterAI.cosine_similarity(vec_a, vec_b)
        self.assertIsNotNone(similarity)
        self.assertIsInstance(similarity, float)

    def test_finding_similar_shows_using_embeddings(self):
        mock_embeddings = {'Show1': np.array([1, 2, 3]), 'Show2': np.array([4, 5, 6])}
        avg_vector = np.array([1.5, 2.5, 3.5])
        result = ShowSuggesterAI.find_similar_shows(mock_embeddings, avg_vector, ['Show2'])
        self.assertIn('Show1', result[0])
        self.assertGreater(100, result[0][1])
        self.assertLess(90, result[0][1])

    def test_calculate_vector_average(self):
        embeddings_dict = {'Show1': [1, 2, 3], 'Show2': [4, 5, 6]}
        show_list = ['Show1', 'Show2']
        avg_vector = ShowSuggesterAI.calculate_vector_average(show_list, embeddings_dict)
        self.assertIsNotNone(avg_vector)
        self.assertEqual(len(avg_vector), 3)

    def test_printing_top_similar_shows(self):
        recommendations = [('Show1', 95.0), ('Show2', 90.0)]
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ShowSuggesterAI.print_top_similar_shows(recommendations)
            self.assertIn('Show1', fake_out.getvalue())
            self.assertIn('Show2', fake_out.getvalue())

    @patch('webbrowser.open', return_value=True)
    def test_displaying_posters(self, mock_open):
        ShowSuggesterAI.display_posters(['http://example.com/poster1.jpg'])
        mock_open.assert_called_with('http://example.com/poster1.jpg')


if __name__ == '__main__':
    unittest.main()
