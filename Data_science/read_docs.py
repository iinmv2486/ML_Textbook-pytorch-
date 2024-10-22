import os
import pickle

# 현재 파일이 있는 디렉토리의 절대 경로를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'docs.pkl')

# 파일을 읽어서 객체로 변환
try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print("파일을 성공적으로 읽었습니다.")
        print(data)
except FileNotFoundError:
    print(f"{file_path} 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
