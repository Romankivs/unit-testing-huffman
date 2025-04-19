import unittest
from collections import Counter
from huffman_file import HuffmanNode, HuffmanCoding, compress_file, decompress_file
import os
import tempfile
import string
import random

class TestHuffmanCoding(unittest.TestCase):
    def setUp(self):
        self.huffman = HuffmanCoding()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_temp_file(self, content):
        temp_file = os.path.join(self.test_dir, "test.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return temp_file

    def test_make_frequency_dict(self):
        # Basic case
        text = "hello"
        freq_dict = self.huffman.make_frequency_dict(text)
        expected = Counter({'h': 1, 'e': 1, 'l': 2, 'o': 1})
        self.assertEqual(freq_dict, expected)

        # Empty string
        freq_dict = self.huffman.make_frequency_dict("")
        self.assertEqual(freq_dict, Counter())

        # Special characters
        text = "a#$\n\t"
        freq_dict = self.huffman.make_frequency_dict(text)
        expected = Counter({'a': 1, '#': 1, '$': 1, '\n': 1, '\t': 1})
        self.assertEqual(freq_dict, expected)

    def test_make_heap(self):
        freq_dict = Counter({'a': 2, 'b': 1, 'c': 3})
        self.huffman.make_heap(freq_dict)
        self.assertEqual(len(self.huffman.heap), 3)

        # Verify heap property
        nodes = sorted(self.huffman.heap, key=lambda x: x.freq)
        self.assertEqual([n.freq for n in nodes], [1, 2, 3])
        self.assertEqual({n.char for n in nodes}, {'a', 'b', 'c'})

        # Empty frequency dict
        self.huffman = HuffmanCoding()
        self.huffman.make_heap(Counter())
        self.assertEqual(len(self.huffman.heap), 0)

    def test_merge_nodes(self):
        # Normal case
        freq_dict = Counter({'a': 1, 'b': 2, 'c': 3})
        self.huffman.make_heap(freq_dict)
        self.huffman.merge_nodes()
        self.assertEqual(len(self.huffman.heap), 1)
        self.assertEqual(self.huffman.heap[0].freq, 6)

        # Single character
        self.huffman = HuffmanCoding()
        freq_dict = Counter({'a': 1})
        self.huffman.make_heap(freq_dict)
        self.huffman.merge_nodes()
        self.assertEqual(len(self.huffman.heap), 1)
        self.assertEqual(self.huffman.heap[0].freq, 1)

        # Empty heap
        self.huffman = HuffmanCoding()
        self.huffman.merge_nodes()
        self.assertEqual(len(self.huffman.heap), 0)

    def test_make_codes(self):
        # Normal case
        text = "aab"
        freq_dict = Counter(text)
        self.huffman.make_heap(freq_dict)
        self.huffman.merge_nodes()
        self.huffman.make_codes()
        self.assertEqual(len(self.huffman.codes), 2)
        self.assertTrue(all(len(code) > 0 for code in self.huffman.codes.values()))
        self.assertEqual(set(self.huffman.reverse_mapping.values()), {'a', 'b'})

        # Single character
        self.huffman = HuffmanCoding()
        freq_dict = Counter({'a': 1})
        self.huffman.make_heap(freq_dict)
        self.huffman.merge_nodes()
        self.huffman.make_codes()
        self.assertEqual(self.huffman.codes, {'a': '0'})
        self.assertEqual(self.huffman.reverse_mapping, {'0': 'a'})

    def test_get_encoded_text(self):
        text = "hello"
        compressed, codes = self.huffman.compress(text)
        encoded = self.huffman.get_encoded_text(text)
        self.assertTrue(all(bit in '01' for bit in encoded))
        self.assertEqual(len(encoded), sum(len(codes[c]) for c in text))

        # Empty text
        self.huffman = HuffmanCoding()
        with self.assertRaises(KeyError):
            self.huffman.get_encoded_text("")

    def test_pad_encoded_text(self):
        # Normal case
        encoded = "1011"  # 4 bits, needs 4 bits padding
        padded = self.huffman.pad_encoded_text(encoded)
        self.assertEqual(len(padded) % 8, 0)
        padding_amount = int(padded[:8], 2)
        self.assertEqual(padding_amount, 4)
        self.assertEqual(padded[8:-padding_amount], encoded)

        # Exact multiple of 8
        encoded = "10101010"  # 8 bits
        padded = self.huffman.pad_encoded_text(encoded)
        self.assertEqual(padded[:8], '00000000')  # No padding needed
        self.assertEqual(padded[8:], encoded)

    def test_get_byte_array(self):
        padded_text = "0000010010110000"  # 2 bytes
        byte_array = self.huffman.get_byte_array(padded_text)
        self.assertEqual(len(byte_array), 2)
        self.assertEqual(bin(byte_array[0])[2:].zfill(8), padded_text[:8])
        self.assertEqual(bin(byte_array[1])[2:].zfill(8), padded_text[8:])

        # Empty string
        with self.assertRaises(ValueError):
            self.huffman.get_byte_array("")

    def test_compress_decompress(self):
        # Normal text
        text = "hello world"
        compressed, codes = self.huffman.compress(text)
        self.huffman.codes = codes
        self.huffman.reverse_mapping = {v: k for k, v in codes.items()}
        decompressed = self.huffman.decompress(compressed)
        self.assertEqual(decompressed, text)

        # Single character repeated
        text = "aaaaaa"
        self.huffman = HuffmanCoding()
        compressed, codes = self.huffman.compress(text)
        self.huffman.codes = codes
        self.huffman.reverse_mapping = {v: k for k, v in codes.items()}
        decompressed = self.huffman.decompress(compressed)
        self.assertEqual(decompressed, text)

        # Special characters
        text = "hello\n\t#$"
        self.huffman = HuffmanCoding()
        compressed, codes = self.huffman.compress(text)
        self.huffman.codes = codes
        self.huffman.reverse_mapping = {v: k for k, v in codes.items()}
        decompressed = self.huffman.decompress(compressed)
        self.assertEqual(decompressed, text)

    def test_file_operations(self):
        # Normal file compression/decompression
        text = "This is a test file for Huffman coding."
        input_file = self.create_temp_file(text)
        
        compress_file(input_file, self.test_dir)
        compressed_file = os.path.join(self.test_dir, "test.huffman")
        codes_file = os.path.join(self.test_dir, "test.codes")
        output_file = os.path.join(self.test_dir, "test_decompressed.txt")
        
        self.assertTrue(os.path.exists(compressed_file))
        self.assertTrue(os.path.exists(codes_file))
        
        success = decompress_file(compressed_file, codes_file, output_file)
        self.assertTrue(success)
        
        with open(output_file, 'r', encoding='utf-8') as f:
            decompressed_text = f.read()
        self.assertEqual(decompressed_text, text)

    def test_empty_file(self):
        # Test compression of empty file
        input_file = self.create_temp_file("")
        with self.assertRaises(ValueError):  # Should handle empty input gracefully
            compress_file(input_file, self.test_dir)

    def test_large_file(self):
        # Test with large random text
        random.seed(42)
        text = ''.join(random.choices(string.ascii_letters + string.digits, k=10000))
        input_file = self.create_temp_file(text)
        
        compress_file(input_file, self.test_dir)
        compressed_file = os.path.join(self.test_dir, "test.huffman")
        codes_file = os.path.join(self.test_dir, "test.codes")
        output_file = os.path.join(self.test_dir, "test_decompressed.txt")
        
        success = decompress_file(compressed_file, codes_file, output_file)
        self.assertTrue(success)
        
        with open(output_file, 'r', encoding='utf-8') as f:
            decompressed_text = f.read()
        self.assertEqual(decompressed_text, text)

    def test_invalid_file_paths(self):
        with self.assertRaises(SystemExit):
            compress_file("non_existent.txt", self.test_dir)

        with self.assertRaises(SystemExit):
            decompress_file("non_existent.txt", "non_existent.codes")

    def test_corrupted_compressed_data(self):
        # Test decompression with corrupted data
        text = "hello"
        input_file = self.create_temp_file(text)
        compress_file(input_file, self.test_dir)
        
        compressed_file = os.path.join(self.test_dir, "test.huffman")
        codes_file = os.path.join(self.test_dir, "test.codes")
        output_file = os.path.join(self.test_dir, "test_decompressed.txt")
        
        # Corrupt compressed file
        with open(compressed_file, 'wb') as f:
            f.write(b"corrupted data")
        
        success = decompress_file(compressed_file, codes_file, output_file)
        self.assertFalse(success)

    def test_corrupted_codes_file(self):
        # Test decompression with corrupted codes file
        text = "hello"
        input_file = self.create_temp_file(text)
        compress_file(input_file, self.test_dir)
        
        compressed_file = os.path.join(self.test_dir, "test.huffman")
        codes_file = os.path.join(self.test_dir, "test.codes")
        output_file = os.path.join(self.test_dir, "test_decompressed.txt")
        
        # Corrupt codes file
        with open(codes_file, 'w', encoding='utf-8') as f:
            f.write("invalid,format\n")
        
        success = decompress_file(compressed_file, codes_file, output_file)
        self.assertFalse(success)

    def test_huffman_node_comparison(self):
        # Test HuffmanNode comparison
        node1 = HuffmanNode('a', 1)
        node2 = HuffmanNode('b', 2)
        node3 = HuffmanNode('c', 1)
        
        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)
        self.assertFalse(node1 < node3)  # Equal frequencies

if __name__ == '__main__':
    unittest.main()
