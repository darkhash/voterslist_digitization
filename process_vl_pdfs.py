# from lib2to3.pgen2.pgen import ParserGenerator
from pdf2image import convert_from_path

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import json
import cv2 
import os

class VotersListProcessor:

    def __init__(self) -> None:
        self.index = 1
        self.counter = 2
        self.out_file = 'default.txt'
        self.error_log = 'error.log'
        self.output_path = 'supp'
        self.temp_path = 'temp'
        self.info_box_thr = 100
        self.image_width  = 0 
        self.image_height = 0


    def process_pdf_dir(self, input_path):
        """Process the pdf files in the given directory

        Args:
            input_path (str): Input path
        """
        # last = 'S10A140P212.pdf'
        # start = False
        for (root, dirs, file) in os.walk(input_path):
            for f in file:
                if '.pdf' in f:
                    # if not start and f != last:
                    #     continue
                    # elif not start:
                    #     start = True
                    print(f'processing: {f}')
                    self.convert_pdf2images(root, f)
            pass

    def convert_pdf2images(self, path, file_name):
        """Extract page images from pdf

        Args:
            path (str): Directory path
            file_name (str): File name of the page image 
        """
        print(f'Extracting images from: {file_name}')
        os.makedirs(os.path.join(self.output_path, path.split('/')[-1]),exist_ok=True)
        self.out_file = os.path.join(self.output_path, path.split('/')[-1],file_name.split('.pdf')[0] + '.txt')
        images = convert_from_path(os.path.join(path, file_name), 300)
        print(f'Number of pages: {len(images)}')
        
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        # pages = [x for x in range(2, len(images)-1)]
        for i in range(2, len(images)-1): #first two pages and last page dont have voter data
            if(self.index - 1 != 30) and i != 2:
                #if the items extracted from a page is less than 30 add that to the error log
                with open('error.log', 'a') as error_log:
                    error_log.write(f'file: {self.out_file}, page: {i-1}, items: {self.index - 1}\n')

            self.index = 1
            # Save pages as images in the pdf
            print(f"saving: {'page'+ str(i) +'.jpg'}")
            images[i].save(os.path.join(self.temp_path,'page'+ str(i) +'.jpg'), 'JPEG') 
            print(f"Processing: {'page'+ str(i) +'.jpg'}")
            try:
                self.image_horizontal_crop(i)
            except Exception as e:
                print(f'Error while processing page: {str(i)}')
                with open(self.error_log, 'a') as error_log:
                        error_log.write(f'Error while processing file: {file_name}, page: {str(i)}\n')
        if(self.index - 1 != 30): #check for the last page 
                with open(self.error_log, 'a') as error_log:
                    error_log.write(f'file: {self.out_file}, page: {len(images) -2}, items: {self.index - 1}\n')
        pass
    
    def image_horizontal_crop(self, page):
        """crop the image into horizontal segments 

        Args:
            page (int): page number
        """
        img = self.top_bottom_crop(os.path.join(self.temp_path,'page'+ str(page) +'.jpg')) # crop the top and bottom portions outside ROI
        self.info_box_thr = np.floor((img.shape[0] //11) * 0.95)
        thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
        img_bin2 = 255-img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh1,img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//100))
        eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
        vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
        # plt.subplot(152),plt.imshow(vertical_lines, cmap = 'gray')
        # plt.title('Image after dilation with vertical kernel'), plt.xticks([]), plt.yticks([])
        # plt.show()  
        # plt.figure(figsize= (30,30))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//100, 1))
        horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=5)
        # plt.subplot(153),plt.imshow(horizontal_lines, cmap = 'gray')
        # plt.title('Image after erosion with horizontal kernel'), plt.xticks([]), plt.yticks([])
        horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)
        # plt.subplot(154),plt.imshow(horizontal_lines, cmap = 'gray')
        # plt.title('Image after dilation with horizontal kernel'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # img_bin = 255-img_bin
        # plotting = plt.imshow(img_bin,cmap='gray')
        # plt.title("Inverted Image with global thresh holding")
        # plt.show()
        # plt.figure(figsize= (30,30))
        # print('hr line:')
        # print(horizontal_lines)
        # test = np.where(horizontal_lines != 0)
        # print([x for x in test])
        # print(test[0][0], test[1][1])
        # return
        vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
        # plt.subplot(151),plt.imshow(vertical_horizontal_lines, cmap = 'gray')
        # plt.title('Erosion'), plt.xticks([]), plt.yticks([])
        thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # plt.subplot(152),plt.imshow(vertical_horizontal_lines, cmap = 'gray')
        # plt.title('global and otsu thresholding'), plt.xticks([]), plt.yticks([])
        # bitxor = cv2.bitwise_xor(img,vertical_horizontal_lines)
        # plt.subplot(153),plt.imshow(bitxor, cmap = 'gray')
        # plt.title('Horizontal and vertical lines image bitxor'), plt.xticks([]), plt.yticks([])
        # bitnot = cv2.bitwise_not(bitxor)
        # plt.subplot(154),plt.imshow(bitnot, cmap = 'gray')
        # plt.title('Horizontal and vertical lines image with bitnot'), plt.xticks([]), plt.yticks([])
        # plt.show()
        img_thr = 255 - vertical_horizontal_lines
        # print(np.floor(img_thr.shape[1]*0.75))
        thr_y = np.floor(img_thr.shape[1]*0.75)
        y_sum = np.count_nonzero(img_thr, axis=1)
        peaks = np.where(y_sum > thr_y)[0]
        peaks = np.concatenate(([0], peaks), axis=0)
        # print(peaks)
        thr_x = 50
        temp = np.diff(peaks).squeeze()
        idx = np.where(temp > thr_x)[0]
        # print(idx)
        peaks = np.concatenate(([0], peaks[idx+1]), axis=0) + 1
        # peaks = np.concatenate(( peaks[idx+1], [0]), axis=0) 
        # print(f'peak shape: {peaks.shape}')
        # print(peaks)
        # cv2.imwrite(f'temp/page{self.counter}_cropped.jpg', img[ peaks[1]:peaks[-1], :] )
        for i in np.arange(peaks.shape[0] - 1):
            # print(peaks[i], peaks[i+1])
            # crop_vertically(img[ peaks[i]:peaks[i+1], :], i)
            # cv2.imwrite('new_image_' + str(i) + '.png', img[ peaks[i]:peaks[i+1], :] )
            image = self.final_left_right_crop(img[ peaks[i]:peaks[i+1], :])
            self.final_row_cut( image, page)
            pass

        # cv2.imwrite(f'new_image_{peaks.shape[0] - 1}.png', img[ peaks[peaks.shape[0] - 1]:, :] )
        image = self.final_left_right_crop(img[ peaks[peaks.shape[0] - 1]:, :])
        self.final_row_cut( image, page )

        img_thr = 255 - vertical_horizontal_lines
        # print(np.floor(img_thr.shape[1]*0.75))
        thr_y = np.floor(img_thr.shape[1]*0.05)
        y_sum = np.count_nonzero(img_thr, axis=1)
        peaks = np.where(y_sum > thr_y)[0]
        peaks = np.concatenate(([0], peaks), axis=0)
        # print(peaks)
        thr_x = 5
        temp = np.diff(peaks).squeeze()
        idx = np.where(temp > thr_x)[0]
        # print(idx)
        peaks = np.concatenate(([0], peaks[idx+1]), axis=0) + 1
        # peaks = np.concatenate(( peaks[idx+1], [0]), axis=0) 
        # print(f'peak shape: {peaks.shape}')
        # print(peaks)
        # cv2.imwrite(f'temp/page{self.counter}_cropped.jpg', img[ peaks[1]:peaks[-1], :] )
        self.counter += 1
        return img[ peaks[1]:peaks[-1], :]
        # self.counter += 1
        pass
    def final_left_right_crop(self, image):
        # img = cv2.imread(image_path,0)
        thresh,img_bin = cv2.threshold(image,128,255,cv2.THRESH_BINARY_INV)
        img_bin2 = 255-image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # print(kernel)
        thresh1,img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
        # import numpy as np
        # plt.figure(figsize= (30,30))

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(image).shape[1]//100))
        eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
        # plt.subplot(151),plt.imshow(eroded_image, cmap = 'gray')
        # plt.title('Image after erosion with vertical kernel'), plt.xticks([]), plt.yticks([])

        vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(image).shape[1]//100, 1))
        horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=5)
        horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)
        vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
        thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_thr = 255 - vertical_horizontal_lines
        thr_y = np.floor(img_thr.shape[0]*0.10)
        y_sum = np.count_nonzero(img_thr, axis=0)
        peaks = np.where(y_sum > thr_y)[0]
        peaks = np.concatenate(([0], peaks), axis=0)
        # print(peaks)
        thr_x = 5
        temp = np.diff(peaks).squeeze()
        idx = np.where(temp > thr_x)[0]
        # print(idx)
        peaks = np.concatenate(([0], peaks[idx+1]), axis=0) + 1
        # peaks = np.concatenate(( peaks[idx+1], [0]), axis=0) 
        # print(f'peak shape: {peaks.shape}')
        # print(peaks)
        # cv2.imshow('lr cropped',image[ :, peaks[1]:peaks[-1]] )
        # cv2.waitKey(0)
        # cv2.imwrite(f'temp/page{self.counter}_lr_cropped.jpg', image[ :, peaks[1]:peaks[-1]] )
        self.counter += 1
        return image[ :, peaks[1]:peaks[-1]]

        pass
    def top_bottom_crop(self, image_path):
        img = cv2.imread(image_path,0)
        self.image_width = img.shape[1]
        thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
        img_bin2 = 255-img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # print(kernel)
        thresh1,img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
        # import numpy as np
        # plt.figure(figsize= (30,30))

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//100))
        eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
        # plt.subplot(151),plt.imshow(eroded_image, cmap = 'gray')
        # plt.title('Image after erosion with vertical kernel'), plt.xticks([]), plt.yticks([])

        vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
        # plt.subplot(152),plt.imshow(vertical_lines, cmap = 'gray')
        # plt.title('Image after dilation with vertical kernel'), plt.xticks([]), plt.yticks([])

        # plt.show()  
        # plt.figure(figsize= (30,30))

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//100, 1))
        horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=5)
        # plt.subplot(153),plt.imshow(horizontal_lines, cmap = 'gray')
        # plt.title('Image after erosion with horizontal kernel'), plt.xticks([]), plt.yticks([])

        horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)
        # plt.subplot(154),plt.imshow(horizontal_lines, cmap = 'gray')
        # plt.title('Image after dilation with horizontal kernel'), plt.xticks([]), plt.yticks([])

        # plt.show()
        # img_bin = 255-img_bin
        # plotting = plt.imshow(img_bin,cmap='gray')
        # plt.title("Inverted Image with global thresh holding")
        # plt.show()
        # plt.figure(figsize= (30,30))
        # print('hr line:')
        # print(horizontal_lines)
        # test = np.where(horizontal_lines != 0)
        # print([x for x in test])
        # print(test[0][0], test[1][1])
        # return

        vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
        # plt.subplot(151),plt.imshow(vertical_horizontal_lines, cmap = 'gray')
        # plt.title('Erosion'), plt.xticks([]), plt.yticks([])

        thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # plt.subplot(152),plt.imshow(vertical_horizontal_lines, cmap = 'gray')
        # plt.title('global and otsu thresholding'), plt.xticks([]), plt.yticks([])

        # bitxor = cv2.bitwise_xor(img,vertical_horizontal_lines)
        # plt.subplot(153),plt.imshow(bitxor, cmap = 'gray')
        # plt.title('Horizontal and vertical lines image bitxor'), plt.xticks([]), plt.yticks([])

        # bitnot = cv2.bitwise_not(bitxor)
        # plt.subplot(154),plt.imshow(bitnot, cmap = 'gray')
        # plt.title('Horizontal and vertical lines image with bitnot'), plt.xticks([]), plt.yticks([])

        # plt.show()
        img_thr = 255 - vertical_horizontal_lines
        # print(np.floor(img_thr.shape[1]*0.75))
        thr_y = np.floor(img_thr.shape[1]*0.05)
        y_sum = np.count_nonzero(img_thr, axis=1)
        peaks = np.where(y_sum > thr_y)[0]
        peaks = np.concatenate(([0], peaks), axis=0)
        # print(peaks)
        thr_x = 5
        temp = np.diff(peaks).squeeze()
        idx = np.where(temp > thr_x)[0]
        # print(idx)
        peaks = np.concatenate(([0], peaks[idx+1]), axis=0) + 1
        # peaks = np.concatenate(( peaks[idx+1], [0]), axis=0) 
        # print(f'peak shape: {peaks.shape}')
        # print(peaks)
        # cv2.imwrite(f'temp/page{self.counter}_cropped.jpg', img[ peaks[1]:peaks[-1], :] )
        self.counter += 1
        return img[ peaks[1]:peaks[-1], :]
    def final_row_cut(self, image, page):
        one_box_width = self.image_width * 0.9 // 3
        print(f'one box width: {one_box_width}')
        if image.shape[1] > 3* one_box_width:
            print(f'Width of three: {image.shape[0]}')
            cut_len = image.shape[1] // 3
            self.box_left_right_crop(image[ :, 0:cut_len], page)
            cv2.imwrite(f'1_sub_image_1.png', 255-image[ :, 0:cut_len])
            self.box_left_right_crop(image[ :, cut_len:2*cut_len], page)
            cv2.imwrite(f'2_sub_image_2.png', 255-image[ :, cut_len:2*cut_len])
            self.box_left_right_crop(image[ :, 2*cut_len:], page)
            cv2.imwrite(f'3_sub_image_3.png', 255-image[ :, 2*cut_len:])
        elif image.shape[1] > 2* one_box_width:
            with open(self.error_log, 'a') as error_log:
                        error_log.write(f'Double: page: {page}, index: {self.index}\n')
            print(f'Width of two: {image.shape[0]}')
            cut_len = image.shape[1] // 2
            self.box_left_right_crop(image[ :, 0:cut_len], page)
            cv2.imwrite(f'1_sub_image_1.png', 255-image[ :, 0:cut_len])
            self.box_left_right_crop(image[ :, cut_len:], page)
            cv2.imwrite(f'2_sub_image_2.png', 255-image[ :, cut_len:])
        else:
            with open(self.error_log, 'a') as error_log:
                        error_log.write(f'Single: page: {page}, index: {self.index}\n')
            print(f'Width of one: {image.shape[0]}')
            self.box_left_right_crop(image, page)
            cv2.imwrite(f'1_sub_image_1.png', 255-image)


    def image_vertical_crop(self, image, page):
        # print(page)
        # print('vertical cropping')
        # Only needed for web reading images

        # Web read image via scikit-image; convert to OpenCV's BGR color ordering
        # img = cv2.cvtColor(io.imread('/Users/hashsingularity/Workspace/datascience/page2.jpg'), cv2.COLOR_RGB2BGR)
       
        # # Inverse binary threshold grayscale version of image
        # img_thr = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY_INV)[1]
        # print(f'image size  {np.shape(img_thr)}')
        img_thr = 255 - image
        # print(np.sort(img_thr))
        # Count pixels along the y-axis, find peaks
        thr_y = np.floor(img_thr.shape[0] * 0.92)
        y_sum = np.count_nonzero(img_thr, axis=0)
        peaks = np.where(y_sum > thr_y)[0]
        peaks = np.concatenate(([0], peaks), axis=0)
        thr_x = 50
        temp = np.diff(peaks).squeeze()
        idx = np.where(temp > thr_x)[0]
        peaks = np.concatenate(([0], peaks[idx+1]), axis=0) + 1

        for i in np.arange(0, peaks.shape[0]-1):
            # print(peaks[i], peaks[i+1])
            # cv2.imwrite('sub_image_' + str(i) + '.png', img[:, peaks[i]:peaks[i+1]])
            self.box_left_right_crop(image[ :, peaks[i]:peaks[i+1]], page)
            # self.convert_to_string(255-img_thr[ :, peaks[i]:peaks[i+1]])
            cv2.imwrite(f'{i}_sub_image_' + str(i) + '.png', 255-img_thr[ :, peaks[i]:peaks[i+1]])
        pass
    def box_left_right_crop(self, img_thr, page):
        """Crop the data rectangle into left and right parts 

        Args:
            img_thr (arr): image fragment
            page (int): page number
        """
        # print(page, idx)
        # print(f'lr: {img_thr.shape}')
        # cv2.imshow('original', img_thr)
        # cv2.waitKey(0)
        img_thr = 255- img_thr
        thr_y = np.floor(img_thr.shape[0]*0.10)
        #y_sum = np.count_nonzero(img_thr, axis=0)
        y_sum = np.count_nonzero(img_thr, axis=0)
        
        # print(f'sum len: {len(y_sum)} shape: {y_sum.shape}')
        # ind = np.argpartition(y_sum, -4)[-4:]
        # print(ind)
        # print('sum')
        # print([x for x in y_sum])
        min = y_sum.min()
        # print(f'Y sum min: {min}')
        # peaks = np.where(y_sum > thr_y)[0]
        peaks = np.where(y_sum < min + 20)[0]
        # print('focus here ----------')
        # print(len(peaks))
        
        # print('printing peaks')
        peaks = np.concatenate(([0], peaks), axis=0)
        # print(peaks)
        longest_seq = max(np.split(peaks, np.where(np.diff(peaks) != 1)[0]+1), key=len).tolist()    
        # print(longest_seq)
        cropped = img_thr[:, 0:(longest_seq[0] + 50)]
        # cv2.imshow('cropped', cropped)
        # cv2.waitKey(0)
        self.convert_to_string(255 - img_thr[:, (longest_seq[0] + 50): -1], lang='eng',page=page)
        self.convert_to_string(255-img_thr[:, 0:(longest_seq[0] + 50)], lang='kan', page=page)
        
        pass

    def convert_to_string(self, img, lang='eng', page=0):
        """Extract text from the image segment using tesseract

        Args:
            img (arr): image fragment
            lang (str, optional): language string for tesseract. Defaults to 'eng'.
            page (int, optional): page number. Defaults to 0.
        """
        try:
            # print(image_path)
            # img = cv2.imread(image_path)

            # Adding custom options
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, lang=lang,config=custom_config)
            # print(text)
            with open(self.out_file, 'a', encoding='utf-8') as out_file:
                text = [line for line in text.split('\n') if line.strip() != '']
                # for line in text:
                
                if lang=='eng':
                    print(f'"page": "{page}", "idx":{self.index}')
                    out_file.write('{')
                    out_file.write(f'"page": "{str(page)}", "idx":"{self.index}"')
                    out_file.write(f',"vid":{json.dumps(text[0])}')
                    # out_file.write('\n')
                    print(self.out_file)
                    # out_file.write('*---\n')
                    print(text)
                    self.index += 1
                else:
                    
                    out_file.write(f',"rows":{json.dumps(text, ensure_ascii=False)}')
                    # out_file.write('\n')
                    out_file.write('}\n')
        except Exception as e:
            print(f'Error: {e}')
            with open(self.error_log, 'a') as erro_log:
                erro_log.write(f'Error while processing: page: {page}, index: {self.index}\n')
            # print('Text: {}')

                

if __name__ == '__main__':
    processor = VotersListProcessor()
    # processor.process_pdf_dir('/Volumes/freestar/ceo.karnataka.gov.in/finalroll_2022/Kannada/MR/AC140')
    processor.process_pdf_dir('./data')
    # with open('/Users/hashsingularity/Workspace/datascience/output/AC001/S10A1P1.txt') as data_file:
    #     for line in data_file:
    #         print(json.loads(line))