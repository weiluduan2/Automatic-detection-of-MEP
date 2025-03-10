import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
import networkx as nx
import pickle
import io
import numpy as np
import torch.nn.functional as F
from Model import Network

class ScaleDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))

            model_path = "../Model/best_model.pth"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Network()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

       # checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')
        return self.transform(image).unsqueeze(0)

    def load_and_preprocess_graph(self, graph_path):
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        return self.graph_to_tensor(graph)

    def graph_to_tensor(self, graph):
        matrix = np.ones((128, 128))
        pos = {node: (node.X, node.Y) for node in graph.nodes()}

        if not pos:
            return torch.ones((1, 1, 128, 128))

        min_x = min(pos[node][0] for node in graph.nodes())
        max_x = max(pos[node][0] for node in graph.nodes())
        min_y = min(pos[node][1] for node in graph.nodes())
        max_y = max(pos[node][1] for node in graph.nodes())

        def normalize(value, min_val, max_val):
            if max_val == min_val:
                return 0.5
            return (value - min_val) / (max_val - min_val)

        for edge in graph.edges():
            start, end = edge
            start_x = int(normalize(pos[start][0], min_x, max_x) * 123) + 2
            start_y = int((1 - normalize(pos[start][1], min_y, max_y)) * 123) + 2
            end_x = int(normalize(pos[end][0], min_x, max_x) * 123) + 2
            end_y = int((1 - normalize(pos[end][1], min_y, max_y)) * 123) + 2

            num_points = int(np.hypot(end_x - start_x, end_y - start_y) * 2)
            points = np.linspace(0, 1, max(num_points, 50))

            for t in points:
                x = int(start_x * (1 - t) + end_x * t)
                y = int(start_y * (1 - t) + end_y * t)
                matrix[y, x] = 0

        for node in graph.nodes():
            x = int(normalize(pos[node][0], min_x, max_x) * 123) + 2
            y = int((1 - normalize(pos[node][1], min_y, max_y)) * 123) + 2
            matrix[y, x] = 0

        matrix = matrix[2:126, 2:126]
        return torch.tensor(matrix, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    def graph_to_image(self, graph):
        plt.figure(figsize=(6.4, 6.4), dpi=100)
        pos = {node: (node.X, node.Y) for node in graph.nodes()}
        nx.draw(graph, pos, node_size=20, node_color='black', with_labels=False)
        plt.axis('off')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
        img_buffer.seek(0)
        img = Image.open(img_buffer).convert('L')
        plt.close()

        return img, self.transform(img).unsqueeze(0)

    def run(self, input1, input2, mode='graph', visualize=False, output_path=None,return_sim=False):
        start_time = time.time()

        if mode == 'graph':
            tensor1 = self.graph_to_tensor(input1)
            tensor2 = self.graph_to_tensor(input2)
        elif mode == 'img':
            if isinstance(input1, str) and isinstance(input2, str):
                tensor1 = self.load_and_preprocess_image(input1)
                tensor2 = self.load_and_preprocess_image(input2)
            elif isinstance(input1, Image.Image) and isinstance(input2, Image.Image):
                tensor1 = self.transform(input1).unsqueeze(0)
                tensor2 = self.transform(input2).unsqueeze(0)
            else:
                raise ValueError("For 'img' mode, input should be image paths or PIL Image objects")

        tensor1 = tensor1.to(self.device)
        tensor2 = tensor2.to(self.device)

        with torch.no_grad():
            logits, similarity = self.model(tensor1, tensor2)
            print("logits:",logits)
            print("Sim is:",similarity)
            probabilities = F.softmax(logits, dim=1)
            prediction = 1 if probabilities[0][1] > 0.5 else 0

        if visualize:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            if mode == 'graph':
                img1, _ = self.graph_to_image(input1)
                img2, _ = self.graph_to_image(input2)
            else:
                img1 = input1 if isinstance(input1, Image.Image) else Image.open(input1)
                img2 = input2 if isinstance(input2, Image.Image) else Image.open(input2)

            ax1.imshow(img1, cmap='gray')
            ax1.set_title("Input 1")
            ax1.axis('off')

            ax2.imshow(img2, cmap='gray')
            ax2.set_title("Input 2")
            ax2.axis('off')

            plt.suptitle(f"Prediction: {'Similar' if prediction == 1 else 'Different'}\n"
                        f"Similarity Score: {probabilities[0][1]:.4f}", fontsize=16)

            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()
            plt.close()

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        print(f"Similarity Score: {probabilities[0][1]:.4f}")
        if return_sim==True:
            return prediction,similarity
        else:
            return prediction

    def test(self, annotation_file, data_dir, mode='img'):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        all_predictions = []
        all_labels = []
        processing_times_graph = []
        processing_times_img = []

        total_start_time = time.time()

        for idx, item in enumerate(annotations):
            try:
                sub_path = os.path.join(data_dir, item['sub_graph'])
            except:
                sub_path = os.path.join(data_dir, item['transform'])
            try:
                standard_path = os.path.join(data_dir, item['tem_graph'])
            except:
                standard_path = os.path.join(data_dir, item['standard_graph'])

            with open(sub_path, 'rb') as f:
                sub_graph = pickle.load(f)
            with open(standard_path, 'rb') as f:
                standard_graph = pickle.load(f)

            start_time_graph = time.time()
            sub_input_direct = self.graph_to_tensor(sub_graph)
            standard_input_direct = self.graph_to_tensor(standard_graph)
            end_time_graph = time.time()
            processing_time_graph = end_time_graph - start_time_graph

            start_time_img = time.time()
            _, sub_input_via_img = self.graph_to_image(sub_graph)
            _, standard_input_via_img = self.graph_to_image(standard_graph)
            end_time_img = time.time()
            processing_time_img = end_time_img - start_time_img

            processing_times_graph.append(processing_time_graph)
            processing_times_img.append(processing_time_img)

            if idx < 5:
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                axes[0, 0].imshow(sub_input_direct.squeeze(), cmap='gray')
                axes[0, 0].set_title("Sub Graph (Direct)")
                axes[0, 1].imshow(standard_input_direct.squeeze(), cmap='gray')
                axes[0, 1].set_title("Standard Graph (Direct)")
                axes[1, 0].imshow(sub_input_via_img.squeeze(), cmap='gray')
                axes[1, 0].set_title("Sub Graph (Via Image)")
                axes[1, 1].imshow(standard_input_via_img.squeeze(), cmap='gray')
                axes[1, 1].set_title("Standard Graph (Via Image)")
                plt.tight_layout()
                plt.savefig(f'comparison_{idx}.png')
                plt.close()

            if mode == 'img':
                sub_input = sub_input_via_img
                standard_input = standard_input_via_img
            else:
                sub_input = sub_input_direct
                standard_input = standard_input_direct

            with torch.no_grad():
                logits, similarity = self.model(standard_input.to(self.device),
                                             sub_input.to(self.device))
                probabilities = F.softmax(logits, dim=1)
                prediction = 1 if probabilities[0][1] > 0.5 else 0

            all_predictions.append(prediction)
            try:
                all_labels.append(int(item['class']))
            except:
                all_labels.append(1)

            print(f"Processed {sub_path} and {standard_path}")
            print(f"  Graph processing time: {processing_time_graph:.4f} seconds")
            print(f"  Image processing time: {processing_time_img:.4f} seconds")

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        average_time_graph = sum(processing_times_graph) / len(processing_times_graph)
        average_time_img = sum(processing_times_img) / len(processing_times_img)

        accuracy = sum([1 for p, l in zip(all_predictions, all_labels) if p == l]) / len(all_labels)
        print(f'\nTest Accuracy: {accuracy:.4f}')
        print(f'Total processing time: {total_time:.4f} seconds')
        print(f'Average graph processing time per pair: {average_time_graph:.4f} seconds')
        print(f'Average image processing time per pair: {average_time_img:.4f} seconds')

        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))

        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_predictions))

        return accuracy, all_predictions, all_labels, processing_times_graph, processing_times_img

    def visualize(self, annotation_file, data_dir, output_dir, mode='img'):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

        for idx, item in enumerate(annotations):
            try:
                sub_path = os.path.join(data_dir, item['sub_graph'])
            except:
                sub_path = os.path.join(data_dir, item['transform'])
            try:
                standard_path = os.path.join(data_dir, item['tem_graph'])
            except:
                standard_path = os.path.join(data_dir, item['standard_graph'])

            print(f"Processing {sub_path} and {standard_path}")

            if mode == 'img':
                sub_input = self.load_and_preprocess_image(sub_path)
                standard_input = self.load_and_preprocess_image(standard_path)
                sub_img = Image.open(sub_path)
                standard_img = Image.open(standard_path)
            elif mode == 'graph':
                with open(sub_path, 'rb') as f:
                    sub_graph = pickle.load(f)
                with open(standard_path, 'rb') as f:
                    standard_graph = pickle.load(f)
                sub_input = self.graph_to_tensor(sub_graph)
                standard_input = self.graph_to_tensor(standard_graph)
                sub_img, _ = self.graph_to_image(sub_graph)
                standard_img, _ = self.graph_to_image(standard_graph)

            with torch.no_grad():
                logits, similarity = self.model(standard_input.to(self.device),
                                             sub_input.to(self.device))
                probabilities = F.softmax(logits, dim=1)
                prediction = 1 if probabilities[0][1] > 0.5 else 0
                similarity_score = probabilities[0][1]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.imshow(sub_img, cmap='gray')
            ax1.set_title("Sub Graph", pad=20)
            ax1.axis('off')

            ax2.imshow(standard_img, cmap='gray')
            ax2.set_title("Tem Graph", pad=20)
            ax2.axis('off')

            plt.suptitle(
                f"Prediction: {'Similar' if prediction else 'Different'}\n"
                f"Similarity Score: {similarity_score:.4f}",
                fontsize=16,
                y=0.95
            )

            sub_name = os.path.basename(sub_path)
            standard_name = os.path.basename(standard_path)
            plt.figtext(
                0.5, 0.02,
                f"Files: {sub_name} vs {standard_name}",
                ha='center',
                fontsize=8,
                wrap=True
            )

            output_path = os.path.join(output_dir, f"result_{idx}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

        print(f"Visualization results saved to {output_dir}")

#Usage example
###########################################################################################
##Input: graph(pkl), Output: prediction(1 for same, 0 for different)(Faster)
detector=ScaleDetector()
graph1_path="../Data/chilled_water_pump_3.pkl"

with open(graph1_path, 'rb') as f:
    graph1 = pickle.load(f)
graph2_path="../Data/fan.pkl"
with open(graph2_path, 'rb') as f:
    graph2 = pickle.load(f)

#
result = detector.run(graph1,graph2,visualize=True,mode='graph')

#################################################################################
##Input: image(png), Output: prediction(1 for same, 0 for different)
# detector=ScaleDetector()
# img1_path = "../Data/chilled_water_pump_false.png"
# img2_path="../Data/chilled_water_pump_90.png"
# result = detector.run(img1_path, img2_path, mode='img', visualize=True)
#
