import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
import cv2
import imageio
import glob
import matplotlib.pyplot as plt

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--ct_dir', type=str, default='', help='Directory of rib segmented chest CT')
    parser.add_argument('--ori_dir', type=str, default='', help='Directory of original chest CT')
    parser.add_argument('--infer_dir', type=str, default='')
    return parser.parse_args()

def numbering(ct):
    '''갈비뼈 segmentation CT에서 24개의 갈비뼈 labeling. 각 갈비뼈 region은 1~24의 voxel value를 갖게 되며, labeling 순서는 random이라고 생각해야 함'''
    labeled_ribs, label_num = label(ct, return_num=True) #labeled_ribs -> label된 ct volume, label_num = label 개수
    
    '''Labeling된 24개 갈비뼈에 대한 region proposal'''
    region = regionprops(labeled_ribs, labeled_ribs)

    '''24개 갈비뼈 높이 max값 순 정렬'''
    sorted_regions = sorted(region, key=lambda x: x.bbox[5])

    '''좌우 확인하며 갈비뼈 번호 지정'''
    #흉곽 중앙 좌표 찾기.
    maxs = []
    mins = []
    for region in sorted_regions:
        maxs.append(region.bbox[3])
        mins.append(region.bbox[0])
    maxs = sorted(maxs)
    mins = sorted(mins)
    middle_coord = (maxs[11]+mins[12]) // 2

    rib_info = [] #갈비뼈 번호 리스트. e.g., 'L1', 'R2'...
    rib_image_values = [] #갈비뼈의 voxel value 리스트. rib_list의 0번 index에 L1이 저장되어 있다면, rib_image_values의 0번 index에는 L1 rib의 voxel value가 저장됨(1~24 중 하나)
    num_count=12 #갈비뼈 총 12쌍 카운팅용 인덱스. CT의 axial slice는 밑에서부터 index가 0이니까 12번 rib부터
    flag=0 #좌/우 2쌍 카운팅용 flag

    #좌우 구별, threshold: middle_coord. 
    for q in range(label_num): #label 된 rib 개수만큼 반복
        if sorted_regions[q].bbox[3] < middle_coord: #우측 rib 찾아서 list들에 추가
            rib_info.append('R' + str(num_count))
            rib_image_values.append(sorted_regions[q].intensity_image.max()) #intensity_image.max()로 region의 voxel value 확인
            flag+=1
        elif sorted_regions[q].bbox[3] > middle_coord: #좌측 rib 찾아서 list들에 추가
            rib_info.append('L' + str(num_count))
            rib_image_values.append(sorted_regions[q].intensity_image.max())
            flag+=1
        if flag%2 == 0: #좌우 1쌍 찾았으면 다음 rib으로
            num_count-=1
    
    return labeled_ribs, rib_image_values, middle_coord, rib_info

def get_slice(axial_3ch, labeled_slice, rib_image_values, middle_coord, rib_info):
    #label된 CT에서 특정 axial slice 내 존재하는 rib region들의 pixel values(label numbers) 추출 
    values_in_slice = np.unique(labeled_slice)

    #rib_list에서 index를 찾기위한 딕셔너리 생성
    #key:각 rib region의 pixel value, value: index of two list
    two_list_index = {}
    for rib_value in rib_image_values:
        if rib_value in values_in_slice: #slice의 rib value 확인 후
                two_list_index[rib_value] = rib_image_values.index(rib_value) #인덱스랑 매칭해서 저장

    identified_rib_regions = regionprops(labeled_slice, labeled_slice) #axial slice 내 갈비뼈들 region proposal
    for identified_region in identified_rib_regions:
        if identified_region.bbox[2] < middle_coord: #중앙 좌표보다 좌측에 있으면 right rib
            rib_number_index = two_list_index[identified_region.intensity_image.max()] #rib region의 pixel value 확인 후 two_list_index 딕셔너리에서 rib_info의 index 얻음
            text = rib_info[rib_number_index] #plotting 할 텍스트는 윗줄에서 얻은 인덱스로 rib_info 리스트에서 가져옴
            x = identified_region.bbox[0]-40
            #x = identified_region.bbox[0] - 60
            y = int((identified_region.bbox[1] + identified_region.bbox[3]) / 2) + 15
            font_size = 0.8
            color1 = (0,0,255) 
            thickness = 2
            cv2.putText(axial_3ch, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_size, color1, thickness)
        elif identified_region.bbox[2] > middle_coord: #중앙 좌표보다 우측에 있으면 left rib
            rib_number_index = two_list_index[identified_region.intensity_image.max()]
            text = rib_info[rib_number_index]
            x = identified_region.bbox[2]+10
            #x = identified_region.bbox[2]+ 30
            y = int((identified_region.bbox[1] + identified_region.bbox[3]) / 2) + 15
            font_size = 0.8
            color2 = (0,0,255)
            thickness = 2
            cv2.putText(axial_3ch, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_size, color2 ,thickness)

    return axial_3ch

def main(args):

    window_min = 0
    window_max = 400
    
    togif = [] #gif저장용
    files = os.listdir(args.infer_dir)

    #files 리스트 내 파일들 하나씩 꺼내서 플랏
    for file in tqdm(files):
        '''갈비뼈만 segmentation 된 CT 로드'''
        #ct1 = nib.load(os.path.join(args.ct_dir, file))
        #mask = nib.load('C:\\Users\\Monet\\Desktop\\dataset\\RibSeg\\RibSeg_490_nii\\nii\\RibFrac175-rib-seg.nii.gz')
        #mask_affine = mask.affine
        #mask = mask.get_fdata()

        infer = nib.load(os.path.join(args.infer_dir, file))
        #infer = nib.load('C:\\Users\\Monet\\Desktop\\dataset\\RibSeg\\RibSeg_490_nii\\nii\\RibFrac11-rib-seg.nii.gz')
        infer_affine = infer.affine
        infer = infer.get_fdata()

        '''원본 CT 로드'''
        file = file.replace('rib-seg', 'image') #원본 CT 파일명으로 변경
        #file = file.replace('infer', 'image')
        #file = 'RibFrac11-image.nii.gz'
        original = nib.load(os.path.join(args.ori_dir, file))
        original_affine = original.affine
        original = original.get_fdata()
        original2 = original
        #original = np.clip(original, window_min, window_max)
        original = torch.tensor(original, dtype=torch.float)
        original = (original - window_min + window_max/2) / window_max
        original = torch.clamp(original, 0, 1) * 255
        original = np.array(original, dtype=np.float)

        #original2 = torch.tensor(original2, dtype=torch.float32)
        #original2 = (original2 - 150 + 100) / 200
        #original2 = torch.clamp(original2, 0, 1) * 255
        #original2 = np.array(original2, dtype=np.float32)

        sagittal = np.rot90(np.sum(original, axis=1))
        sagittal = (sagittal - sagittal.min()) / (sagittal.max() - sagittal.min())
        sagittal = (sagittal * 255).astype(np.uint8)
        #plt.subplot(1,3,1)
        #plt.imshow(sagittal, cmap='gray')

        #padd_rows = 512 - original.shape[0]
        #sagittal = np.pad(sagittal, ((0, padd_rows), (0,0)), mode='constant', constant_values=0)

        sagittal = cv2.resize(sagittal, (512,512))
        #plt.subplot(1,3,2)
        #plt.imshow(sagittal,cmap='gray')

        sagittal = sagittal.astype(np.float32)
        #sagittal = cv2.cvtColor(sagittal, cv2.COLOR_GRAY2BGR)

        #sagittal = cv2.cvtColor(sagittal, cv2.COLOR_GRAY2BGR)
        height = original.shape[-1]
        color = (255,255,255)
        thickness = 1
        spacing = 512 // (height // 10)
        previous= (512 - spacing*(height//10)) #//2
        #previous = 0
        line_coords = []
        for i in range(1, height//10 - 1):
            y = spacing + previous
            start_point = (0,y)
            end_point = (511, y)
            line_coords.append(y)
            previous = y
            cv2.line(sagittal, start_point, end_point, color, thickness)

        #plt.subplot(1,3,3)
        #plt.imshow(sagittal, cmap='gray')
        #plt.show()

        #mask_labeled_ribs, mask_rib_values, mask_middle_coord, mask_rib_info = numbering(mask)
        infer_labeled_ribs, infer_rib_values, infer_middle_coord, infer_rib_info = numbering(infer)

        '''axial slice 0번부터 끝번까지 갈비뼈 번호랑 같이 plot'''
        slice_num = original.shape[2]
        i = slice_num-1

        # Create a dotted vertical line image with the same height as the input images
        line_thickness = 2  # Adjust the thickness of the line
        line_color = (0, 0, 255)  # Set the line color (BGR format)
        dot_length = 5  # Length of each dot
        gap_length = 5  # Length of the gap between dots
        vertical_line = np.zeros((512, line_thickness, 3), dtype=np.uint8)
        for j in range(0, 512, dot_length + gap_length):
            vertical_line[j:j+dot_length, :] = line_color
        
        sagittal = np.stack((sagittal,)*3, axis=-1)
        #for i in range(slice_num-1, -1, -1):
        changeline = original.shape[-1] // len(line_coords)
        linecnt = 0
        while i > 0:
            sliceNo = i
            axial_slice = original[:,:,sliceNo] #원본 CT에서 axial slice 추출
            axial_slice = np.expand_dims(axial_slice, axis=-1) #library에서 사용하기 위해 맨앞에 색상 나타내는(grayscale) dimension 추가
            axial_slice = np.rot90(np.rot90(np.rot90(axial_slice))) #보기 편하게 돌리기
            axial_slice = np.ascontiguousarray(axial_slice, dtype=np.uint8)
            axial_slice = cv2.flip(axial_slice, 1) #neurological convention을 radiological convention으로
            #mask_labeled_slice = mask_labeled_ribs[:,:,sliceNo] #label된 CT에서 axial slice 추출
            infer_labeled_slice = infer_labeled_ribs[:,:,sliceNo]
            #mask_axial_3ch = cv2.cvtColor(axial_slice, cv2.COLOR_GRAY2BGR)
            infer_axial_3ch = cv2.cvtColor(axial_slice, cv2.COLOR_GRAY2BGR)
            axial_slice_3ch = cv2.cvtColor(axial_slice, cv2.COLOR_GRAY2BGR)

            #mask_axial_3ch = get_slice(mask_axial_3ch, mask_labeled_slice, mask_rib_values, mask_middle_coord, mask_rib_info)
            infer_axial_3ch = get_slice(infer_axial_3ch, infer_labeled_slice, infer_rib_values, infer_middle_coord, infer_rib_info)

            text = "Original: 3D volume of chest CT"
            cv2.putText(sagittal, text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            #text = "FileName: " + file.replace("-image.nii.gz", "") + " / Slice no."+str(sliceNo)
            #cv2.putText(maks_axial_3ch, text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            text2 = "Input: 2D axial slice / " + "Slice no."+str(sliceNo)
            cv2.putText(axial_slice_3ch, text2, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            text3 = "Output: 2D axial slice with rib numbering / " + "Slice no."+str(sliceNo)
            cv2.putText(infer_axial_3ch, text3, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if linecnt < changeline:
                color = (0,255,0)
                start_point = (0,line_coords[0])
                end_point = (511, line_coords[0])
                cv2.line(sagittal, start_point, end_point, color, thickness)
                linecnt+=1
            else:
                start_point = (0,line_coords[0])
                end_point = (511, line_coords[0])
                cv2.line(sagittal, start_point, end_point, (255,255,255), thickness)

                line_coords.pop(0)
                color = (0,255,0)
                start_point = (0,line_coords[0])
                end_point = (511, line_coords[0])
                cv2.line(sagittal, start_point, end_point, color, thickness)

                linecnt = 0


            sagittal = sagittal.astype(np.uint8)
            #좌: bone window setting된 axial slice / 우: rib info 들어간 axial slice
            #ori_and_numbered = cv2.hconcat([mask_axial_3ch, vertical_line, infer_axial_3ch])
            ori_and_numbered = cv2.hconcat([sagittal, vertical_line, axial_slice_3ch, vertical_line, infer_axial_3ch])
            #ori_and_numbered = cv2.hconcat([sagittal, vertical_line, infer_axial_3ch])

            togif.append(ori_and_numbered)
            i -= 1
            
            #둘 중 하나 선택해서 plot하시면 됩니다!
            '''직접 마우스 클릭으로 창 닫으면서 확인'''
            '''cv2.imshow(file.split("-")[0], ori_and_numbered) #plot창 상단에 띄울 파일명, axial slices
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

            '''1초마다 자동으로 다음 axial slice 띄움'''
            '''cv2.imshow(file.split("-")[0], ori_and_numbered)
            #cv2.waitKey(500) #1000밀리초 = 1초
     
            key = cv2.waitKey(500) & 0xFF
            if key == ord('q'): #종료
                break
            elif key == ord('b') and i < slice_num: #이전슬라이스로
                i += 2 #2개씩
            elif key == ord('s'): #현재슬라이스에서 스탑
                while(1):
                    key = cv2.waitKey() & 0xFF
                    if key == ord('s'): #재시작
                        break
            else: #자동진행
                i -= 1'''

        #cv2.destroyAllWindows()
        
        '''gif save'''
        '''resize_ratio = 0.5
        resized_togif = [cv2.resize(img, (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio))) for img in togif]
        frame_skip = 2
        reduced_togif = resized_togif[::frame_skip]
        #imageio.mimsave(savefilename, reduced_togif, duration=0.1) #0.1초단위로 슬라이스 훑어내려갑니다

        savefilename = file.replace("-image.nii.gz", "") + '.gif'
        imageio.mimsave(savefilename, togif, duration=0.1) #0.1초단위로 슬라이스 훑어내려갑니다'''


        '''mp4 save'''
        # Set video properties
        savefilename = file.replace("-image.nii.gz", "") + '.mp4'
        width, height = togif[0].shape[1], togif[0].shape[0]
        fps = 10  # Change this to the desired frame rate
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # MP4 codec
        bitrate = 10_000_000
        #high_res_frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC) for frame in togif]
        #savefilename = f'output_for_paper_high_quality_bitrate{bitrate}.mp4'
        # Create a VideoWriter object
        video_writer = cv2.VideoWriter(savefilename, fourcc, fps, (width, height), isColor=True)
        #video_writer.set(cv2.VIDEOWRITER_PROP_BITRATE, 10_000_000)

        # Write frames to the video
        for frame in togif:
            video_writer.write(frame)

        # Release the VideoWriter object
        video_writer.release()

        togif.clear()


if __name__ =='__main__':
    args = parse_args()
    main(args)