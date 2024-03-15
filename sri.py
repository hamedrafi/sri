#!/usr/bin/python3

import PIL
import numpy
from PIL import Image
from pathlib import Path
import sqlite3
import numpy as np
import cv2
from cv2 import Mat
from utils import Color, ImageSegment





class ImageDataBase:
    def __init__(self, square_size=40):
        self.image_files = []
        self.square_size = square_size

    def scan_dir(self, image_db_path: str | Path):
        self.image_files = [f for f in image_db_path.iterdir() if f.is_file()]

    @staticmethod
    def find_dominant_color(self, image):
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        dominant_color = np.uint8(avg_color)
        return dominant_color

    # def apply_tone_to_image(self, photo:  cv2.Mat | np.ndarray[Any, np.dtype[np.generic]] | np.ndarray, tone):
    #     cv2.imread("ff")
    #     gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    #     normalized_gray = np.array(gray, np.float32) / 255
    #     # solid color
    #     sepia = np.ones(photo.shape)
    #     sepia[:, :, 0] *= 153  # B
    #     sepia[:, :, 1] *= 204  # G
    #     sepia[:, :, 2] *= 255  # R
    #     # hadamard
    #     sepia[:, :, 0] *= normalized_gray  # B
    #     sepia[:, :, 1] *= normalized_gray  # G
    #     sepia[:, :, 2] *= normalized_gray  # R
    #     return np.array(sepia, np.uint8)

    def insert_images_to_db(self):
        sqlite_connection = sqlite3.connect('SQLite_Python.db')
        for image_path in self.image_files:
            try:
                img = cv2.imread(image_path)

                maxsize = (1024, 1024)
                img = cv2.resize(img, maxsize, interpolation=cv2.INTER_AREA)
                cursor = sqlite_connection.cursor()
                print("Connected to SQLite")
                sqlite_insert_blob_query = """ INSERT INTO new_thumbnail
                                          (photo, dominant_color, width) VALUES (?, ?, ?)"""

                dominant_color = self.find_dominant_color(photo=img)
                # Convert data into tuple format
                data_tuple = (img, dominant_color, img.width)
                cursor.execute(sqlite_insert_blob_query, data_tuple)
                sqlite_connection.commit()
                print("\rImage and file inserted successfully as a BLOB into a table")
                cursor.close()

            except (PIL.UnidentifiedImageError, FileNotFoundError) as err:
                continue
            except sqlite3.Error as error:
                print("Failed to insert blob data into sqlite table", error)
            finally:
                if sqlite_connection:
                    sqlite_connection.close()
                    print("the sqlite connection is closed")
                img.close()

    def segment_and_extract_colors(self, image_path: str | Path):
        """
        This function should receive an image path, reads it and based on the step size it should segment the image
        into squares. It should then get the dominant colors of those squares. Finally it should return with a list
        of dominant colors an their coordinates withing the image size.
        Here the image is 24x12 pixels
        Square Size is set to 3
        Segments are sections with size 3x3

              ╔═════> A Pixel                     ╔═════>  A Square Segment
        ┌───┬─╫─┬───┬───┬───┬───┬───┬───┬───┬─────╫─────┬───────────┬───────────┬───────────┬───────────┐
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┤     ║     │           │           │           │           │
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┤           │           │           │           │           │
        ├───┼───┼───┼───┼───┼───┼───┴───┴───┼───────────┼───────────┼───────────┼───────────┼───────────┤
        ├───┼───┼───┼───┼───┼───┤           │           │           │           │           │           │
        ├───┼───┼───┼───┼───┼───┤           │           │           │           │           │           │
        ├───┼───┼───┼───┴───┴───┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
        ├───┼───┼───┤           │           │           │           │           │           │           │
        ├───┼───┼───┤           │           │           │           │           │           │           │
        ├───┴───┴───┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
        │           │           │           │           │           │           │           │           │
        │           │           │           │           │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

        :param image_path:
        :param square_size:
        :return:
        """
        orig_img = cv2.imread(image_path)
        img = orig_img.copy()
        if img is None:
            print("Error: Image not found.")
            return

        # Determine the dimensions of the checkerboard squares
        height, width, _ = img.shape
        num_squares_height = height // self.square_size
        num_squares_width = width // self.square_size

        # Initialize list to store dominant colors
        dominant_colors = []

        # Iterate over the image and apply checkerboard pattern
        for i in range(num_squares_height):
            row_colors = []  # Store dominant colors for each row
            for j in range(num_squares_width):
                # Extract the current square from the image
                square = img[i * self.square_size:(i + 1) * self.square_size,
                             j * self.square_size:(j + 1) * self.square_size]

                # Calculate the dominant color of the square
                avg_color_per_row = np.average(square, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                dominant_color = np.uint8(avg_color)

                row_colors.append(dominant_color)

                # Apply the dominant color to the square (Optional)
                img[i * self.square_size:(i + 1) * self.square_size,
                    j * self.square_size:(j + 1) * self.square_size] = dominant_color

            dominant_colors.append(row_colors)

        return orig_img, img, dominant_colors

    @staticmethod
    def find_largest_square(image) -> numpy.array:
        """
        Find the largest square crop of an image.

        Parameters:
        image (numpy.ndarray): Input image as a NumPy array.

        Returns:
        numpy.ndarray: Largest square crop of the input image.
        """
        height, width = image.shape[:2]
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        end_x = start_x + min_dim
        end_y = start_y + min_dim
        return image[start_y:end_y, start_x:end_x]

    def apply_dominant_color_tones(self, image, dominant_colors):
        # Create a copy of the original image
        toned_image = self.find_largest_square(image.copy())

        gray = cv2.cvtColor(toned_image, cv2.COLOR_RGB2GRAY)
        normalized_gray = np.array(gray, np.float32) / 255

        """
        We need to allocate memory for the entire resulting image and fill in the same memory block
        rather than stacking memory which requires extra space for resulting arrays
        """
        total_segments = sum(len(v) for v in dominant_colors)



        row_images_segments = []
        # Iterate over the dominant colors and apply them as tones to the entire image
        print("Building final image")
        for row_idx, row_colors in enumerate(dominant_colors):
            column_image_segments = []
            for col_idx, color in enumerate(row_colors):
                toned = np.ones(toned_image.shape)
                toned[:, :, 0] *= color[0]  # R
                toned[:, :, 1] *= color[1]  # G
                toned[:, :, 2] *= color[2]  # B

                toned[:, :, 0] *= normalized_gray  # R
                toned[:, :, 1] *= normalized_gray  # G
                toned[:, :, 2] *= normalized_gray  # B

                new_col_img = np.array(toned, np.uint8)
                column_image_segments.append(new_col_img)

            row_image = np.hstack(column_image_segments)
            row_images_segments.append(row_image)
            print(f"\r {round(row_idx/len(dominant_colors)*100, 2)}% in progress", end="")

        print("\nStacking horizontal images\n")
        final_image = np.vstack(row_images_segments)
        # Save the stacked image
        print("\nSaving final image\n")
        # cv2.imwrite(f"/home/hamed/Pictures/ai/final_{self.square_size}.jpg", final_image)
        small_final = cv2.resize(final_image, (0, 0), fx=0.2, fy=0.2)
        # cv2.imwrite(f"/home/hamed/Pictures/ai/final_{self.square_size}_small.jpg", small_final)


def main():
    """
    This is the features that need to be implemented in the program

    1. scan directory to find all image files
    2. Make a square shaped thumbnail copy of all those images with size that will be used in the final reconstructed image
    3. Make a DB of these images with the following information:
        3.0 The DB can include the thumbnail image itself or just the path to the image.
        3.1 Most dominant color in the thumbnail
        3.2 Thumbnail size (height and width)
    4. Scan the reference image and disintegrate it into smaller square blocks. The block size should be <= thumbnail size
    5. For every block in picture:
        5.1 Find most dominant color of each block
        5.2 Select next available thumbnail with the same marginal dominant color
        5.3 Map block index and position with the selected thumbnail
    6. After all blocks were mapped to a thumbnail, read the thumbnails and concatenate them together into one large
       final image.
    7. Save the new file

    :return:
    """

    imgdb = ImageDataBase(square_size=15)
    original_image, _, dominant_colors_2d = imgdb.segment_and_extract_colors(
        image_path="/home/hamed/Pictures/ai/msg133703258-305819.jpg",
    )

    # Apply dominant color tones to the original image and display them one by one
    imgdb.apply_dominant_color_tones(original_image, dominant_colors_2d)


if __name__ == '__main__':
    main()
