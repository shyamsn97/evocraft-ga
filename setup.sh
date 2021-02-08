git clone https://github.com/real-itu/Evocraft-py.git evocraft_py
python -m pip install -r evocraft-requirements.txt
cp evocraft_py/minecraft_pb2.py .
cp evocraft_py/minecraft_pb2_grpc.py .
cd ./evocraft_py && java -jar spongevanilla-1.12.2-7.3.0.jar
cd .. && python edit_eula_file.py --file_path=evocraft_py/eula.txt
