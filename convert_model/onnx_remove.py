import onnx
from onnx import helper

model = onnx.load("pinet_v2_remove_final.onnx")
graph = model.graph

graph.output.pop(15)
graph.output.pop(14)
graph.output.pop(13)
graph.output.pop(12)
graph.output.pop(8)
graph.output.pop(7)
graph.output.pop(6)
graph.output.pop(5)
graph.output.pop(4)
graph.output.pop(3)
graph.output.pop(2)
graph.output.pop(1)
graph.output.pop(0)
onnx.checker.check_model(model)
onnx.save(model, 'pinet_v2_remove2.onnx')