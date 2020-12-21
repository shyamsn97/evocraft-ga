git clone https://github.com/real-itu/Evocraft-py.git temp
rsync -a temp/* evocraft_ga/external/
rm -rf temp
cd evocraft_ga/external/ && python edit_file.py --file_path=minecraft_pb2_grpc.py --find_str="import minecraft_pb2" --replace_str="import evocraft_ga.external.minecraft_pb2"
cd ../../
python setup.py install