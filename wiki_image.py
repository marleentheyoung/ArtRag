import requests
from PIL import Image
from io import BytesIO
import hashlib

def get_wikimedia_image(filename):
    """
    Fetches an image from Wikimedia Commons given a filename.
    
    Args:
        filename (str): The filename like "Pablo_picasso_1.jpg"
    
    Returns:
        PIL.Image: The downloaded image as a PIL Image object
    
    Raises:
        requests.RequestException: If the HTTP request fails
        ValueError: If the image cannot be processed
    """
    
    # Create MD5 hash of the filename for the URL structure
    md5_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
    
    # Wikimedia Commons URL structure uses first character and first two characters of MD5
    first_char = md5_hash[0]
    first_two_chars = md5_hash[:2]
    
    # Construct the full URL
    url = f"https://upload.wikimedia.org/wikipedia/commons/{first_char}/{first_two_chars}/{filename}"
    
    try:
        # Fetch the image
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Convert to PIL Image
        image = Image.open(BytesIO(response.content))
        
        return image
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch image from {url}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

# Alternative simpler version if you know the exact path structure
def get_wikimedia_image_simple(filename, path="9/98"):
    """
    Simplified version when you know the exact path structure.
    
    Args:
        filename (str): The filename like "Pablo_picasso_1.jpg"
        path (str): The path structure like "9/98"
    
    Returns:
        PIL.Image: The downloaded image as a PIL Image object
    """
    
    url = f"https://upload.wikimedia.org/wikipedia/commons/{path}/{filename}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        return image
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch image from {url}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Using the simple version with known path
        image = get_wikimedia_image_simple("Pablo_picasso_1.jpg", "9/98")
        
        # Display image info
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        
        # Save locally if needed
        # image.save("downloaded_image.jpg")
        
        # Show the image (if running in an environment that supports it)
        # image.show()
        
    except Exception as e:
        print(f"Error: {e}")