import heapq
from collections import Counter
import os

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        
    def make_frequency_dict(self, text):
        return Counter(text)
    
    def make_heap(self, frequency):
        for char, freq in frequency.items():
            node = HuffmanNode(char, freq)
            heapq.heappush(self.heap, node)
    
    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            
            heapq.heappush(self.heap, merged)
    
    def make_codes_helper(self, node, current_code):
        if node is None:
            return
        
        if node.char is not None:
            self.codes[node.char] = current_code or "0"
            self.reverse_mapping[current_code or "0"] = node.char
            return
        
        self.make_codes_helper(node.left, current_code + "0")
        self.make_codes_helper(node.right, current_code + "1")
    
    def make_codes(self):
        if not self.heap:
            return
        root = heapq.heappop(self.heap)
        self.make_codes_helper(root, "")
    
    def get_encoded_text(self, text):
        if not self.codes:
            raise KeyError("No Huffman codes available")
        encoded_text = ""
        for char in text:
            if char not in self.codes:
                raise KeyError(f"Character '{char}' not found in Huffman codes")
            encoded_text += self.codes[char]
        return encoded_text
    
    def pad_encoded_text(self, encoded_text):
        padding_amount = 8 - (len(encoded_text) % 8)
        if padding_amount == 8:
            padding_amount = 0
            
        padding_info = format(padding_amount, '08b')
        padded_text = padding_info + encoded_text + ('0' * padding_amount)
        
        return padded_text
    
    def get_byte_array(self, padded_encoded_text):
        if not padded_encoded_text:
            raise ValueError("Cannot create byte array from empty input")
        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return bytes(b)
    
    def compress(self, text):
        if not text:
            raise ValueError("Cannot compress empty text")
            
        frequency = self.make_frequency_dict(text)
        
        self.make_heap(frequency)
        
        self.merge_nodes()
        
        self.make_codes()
        
        encoded_text = self.get_encoded_text(text)
        padded_encoded_text = self.pad_encoded_text(encoded_text)
        byte_array = self.get_byte_array(padded_encoded_text)
        
        return byte_array, self.codes
    
    def remove_padding(self, bit_string):
        if len(bit_string) < 8:
            raise ValueError("Invalid bit string: too short")
        padding_info = bit_string[:8]
        try:
            padding_amount = int(padding_info, 2)
        except ValueError:
            raise ValueError("Invalid padding information")
        
        if padding_amount > 7 or (padding_amount > 0 and len(bit_string) < 8 + padding_amount):
            raise ValueError("Invalid bit string: invalid padding amount or insufficient length")
        
        bit_string = bit_string[8:-padding_amount] if padding_amount > 0 else bit_string[8:]
        if not bit_string:
            raise ValueError("Invalid bit string: no data after padding")
        
        return bit_string
    
    def decode_text(self, encoded_text):
        if not encoded_text:
            raise ValueError("Invalid encoded text: empty input")
        
        current_code = ""
        decoded_text = ""
        
        for bit in encoded_text:
            if bit not in '01':
                raise ValueError("Invalid encoded text: contains non-binary characters")
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""
        
        if current_code:
            raise ValueError("Invalid encoded text: incomplete code")
                
        return decoded_text
    
    def decompress(self, byte_array):
        bit_string = ""
        for byte in byte_array:
            bits = format(byte, '08b')
            bit_string += bits
        
        try:
            encoded_text = self.remove_padding(bit_string)
            decompressed_text = self.decode_text(encoded_text)
        except ValueError:
            return ""
        
        return decompressed_text

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        title = os.path.basename(file_path)
        return title, content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def save_compressed_data(compressed_data, output_file):
    try:
        with open(output_file, 'wb') as file:
            file.write(compressed_data)
        return True
    except Exception as e:
        print(f"Error saving compressed data: {e}")
        return False

def load_compressed_data(input_file):
    try:
        with open(input_file, 'rb') as file:
            compressed_data = file.read()
        return compressed_data
    except Exception as e:
        print(f"Error loading compressed data: {e}")
        return None

def save_huffman_codes(codes, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for char, code in codes.items():
                char_code = ord(char)
                file.write(f"{char_code},{code}\n")
        return True
    except Exception as e:
        print(f"Error saving Huffman codes: {e}")
        return False

def load_huffman_codes(input_file):
    codes = {}
    reverse_mapping = {}
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    char_code, code = line.split(',')
                    char = chr(int(char_code))
                    codes[char] = code
                    reverse_mapping[code] = char
        return codes, reverse_mapping
    except Exception as e:
        print(f"Error loading Huffman codes: {e}")
        return {}, {}

def print_huffman_code_table(most_common, codes):
    code_table = []
    for char, freq in most_common:
        if char in codes:
            char_display = char if char.isprintable() and char != ' ' else f"'{repr(char)[1:-1]}'"
            code_table.append((char_display, codes[char], freq, len(codes[char])))
    
    print(f"{'Character':<10} {'Code':<20} {'Frequency':<10} {'Bits':<5}")
    print("-" * 45)
    for char, code, freq, bits in code_table:
        print(f"{char:<10} {code:<20} {freq:<10} {bits:<5}")

def compress_file(input_file, output_dir=None):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        raise SystemExit(1)
        
    title, content = read_text_file(input_file)
    if content is None:
        print(f"Error: Failed to read input file '{input_file}'")
        raise SystemExit(1)
    
    if not content:
        raise ValueError("Cannot compress empty file")
    
    if output_dir is None:
        output_dir = "."
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    huffman = HuffmanCoding()
    
    print("Compressing data using Huffman coding...")
    compressed_data, codes = huffman.compress(content)
    
    original_size = len(content.encode('utf-8'))
    compressed_size = len(compressed_data)
    compression_ratio = (original_size - compressed_size) / original_size * 100

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    compressed_file = os.path.join(output_dir, f"{base_name}.huffman")
    codes_file = os.path.join(output_dir, f"{base_name}.codes")
        
    print("Huffman Codes:")
    char_freq = huffman.make_frequency_dict(content)
    most_common = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
    
    print_huffman_code_table(most_common, codes)
    
    if not save_compressed_data(compressed_data, compressed_file):
        raise IOError(f"Failed to save compressed data to {compressed_file}")
    if not save_huffman_codes(codes, codes_file):
        raise IOError(f"Failed to save Huffman codes to {codes_file}")
    
    print(f"\nCompression Statistics:")
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Space saved: {original_size - compressed_size} bytes ({compression_ratio:.2f}%)")

def decompress_file(compressed_file, codes_file, output_file=None):
    if not os.path.exists(compressed_file):
        print(f"Error: Compressed file '{compressed_file}' not found")
        raise SystemExit(1)

    compressed_data = load_compressed_data(compressed_file)
    if compressed_data is None:
        return False
    
    codes, reverse_mapping = load_huffman_codes(codes_file)
    if not codes:
        return False
    
    huffman = HuffmanCoding()
    huffman.codes = codes
    huffman.reverse_mapping = reverse_mapping
    
    decompressed_text = huffman.decompress(compressed_data)
    if not decompressed_text and compressed_data:
        return False
    
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(compressed_file))[0]
        output_file = f"{base_name}_decompressed.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(decompressed_text)
        print(f"Decompressed text saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving decompressed text: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Compress:   python huffman_coding.py compress <input_file> [output_directory]")
        print("  Decompress: python huffman_coding.py decompress <compressed_file> <codes_file> [output_file]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "compress":
        if len(sys.argv) < 3:
            print("Error: Input file path required for compression")
            print("Usage: python huffman_coding.py compress <input_file> [output_directory]")
            sys.exit(1)
            
        input_file = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        
        try:
            compress_file(input_file, output_dir)
        except (ValueError, IOError, SystemExit) as e:
            print(f"Compression failed: {e}")
            sys.exit(1)
        
    elif command == "decompress":
        if len(sys.argv) < 4:
            print("Error: Compressed file and codes file required for decompression")
            print("Usage: python huffman_coding.py decompress <compressed_file> <codes_file> [output_file]")
            sys.exit(1)
            
        compressed_file = sys.argv[2]
        codes_file = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        
        if not os.path.exists(compressed_file):
            print(f"Error: Compressed file '{compressed_file}' not found")
            sys.exit(1)
            
        if not os.path.exists(codes_file):
            print(f"Error: Codes file '{codes_file}' not found")
            sys.exit(1)
            
        success = decompress_file(compressed_file, codes_file, output_file)
        
        if success:
            print("\nDecompression Complete!")
        else:
            print("\nDecompression Failed!")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        print("Usage:")
        print("  Compress:   python huffman_coding.py compress <input_file> [output_directory]")
        print("  Decompress: python huffman_coding.py decompress <compressed_file> <codes_file> [output_file]")
        sys.exit(1)
