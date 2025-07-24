import os
import cairosvg
import argparse

def main(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all SVG files in the input folder
    for svg_file in os.listdir(input_folder):
        if svg_file.endswith('.svg'):  # Check if the file is an SVG
            input_svg_path = os.path.join(input_folder, svg_file)
            
            # Create the output PNG file path
            output_png_path = os.path.join(output_folder, f"{os.path.splitext(svg_file)[0]}.png")
            
            # Use cairosvg to convert SVG to PNG
            try:
                cairosvg.svg2png(url=input_svg_path, write_to=output_png_path)
                print(f"Converted {svg_file} to PNG.")
            except Exception as e:
                print(f"Failed to convert {svg_file}: {e}")

    print("SVG to PNG conversion completed.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert SVG files to PNG format.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing SVG files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the PNG files.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args.input_dir, args.output_dir)
