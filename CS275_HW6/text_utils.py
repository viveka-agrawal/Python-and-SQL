
import numpy as np

def read_text_file(filename):
	"""
    Load a text file, replace all whitespace with underscores, and convert it to a NumPy array of characters.

    Parameters:
    ----------
    filename : str
        Path to the text file.

    Returns:
    -------
    np.ndarray of shape (N,)
        1D array of characters, with whitespace replaced by underscores.
    """

	text_file = open(filename, "r")
	data = text_file.read()
	text_file.close()
	 
	# Convert spaces to underscores
	words = data.split()
	words = '_'.join(words)

	# Convert to numpy
	words = np.asarray(list(words))
	return words


def text_to_num(text):
	"""
    Convert an array of integer character codes back to string:
        - 0 to 25 → 'a' to 'z'
        - 26 → '_'
        - 27 → '*'   (This is the "invalid" or "missing" character)

    Parameters:
    ----------
    nums : np.ndarray or list of int, shape (N,)
        Integer-encoded characters.

    Returns:
    -------
    str
        Reconstructed text string from character codes.
    """

	# convert to list
	text = text.tolist()

	# Convert to ascii
	as_nums = [ord(s) for s in text]

	# Make 'a' == 0
	as_nums = [n - 97 for n in as_nums]

	# Convert underscores to 26
	as_nums = [26 if n==-2 else n for n in as_nums]	

	# Convert '*' to 27
	as_nums = [27 if n==-55 else n for n in as_nums]	

	# Convert to numpy array
	return np.asarray(as_nums)


def num_to_text(nums):
	''' 
		Convert nums to ascii characters
			- 0-25 => a-z
			- 26 => "_"
			- 27 => "*"   (This is the "invalid" or "missing" character)

		Parameters:
			nums: Numbers to convert

		Return:
			numpy array of converted numbers to text
	'''
	nums = nums.tolist()
	nums = [-55 if n==27 else n for n in nums]	
	nums = [-2 if n==26 else n for n in nums]	
	nums = [n+97 for n in nums]	
	return "".join(chr(n) for n in nums)


def sample_hmm_text(model, sample_length=500):
    """
    Sample characters from a trained HMM and return them as a string.

    Parameters:
    ----------
    model : hmmlearn.hmm.CategoricalHMM
        A trained categorical HMM model.
    sample_length : int, optional
        Number of characters to sample. Default is 500.

    Returns:
    -------
    str
        Sampled text string.
    """
    sample, _ = model.sample(sample_length)
    return num_to_text(sample.flatten())


def erase_random_chars(text_array, erase_rate=0.2, erase_char='*'):
    """
    Randomly erase characters in a text array by replacing them with a placeholder character.

    Parameters:
    ----------
    text_array : np.ndarray of shape (N,)
        Original text as a 1D array of characters.
    erase_rate : float, optional
        Probability of each character being erased. Default is 0.2.
    erase_char : str, optional
        Character used to indicate erasure. Default is '*'.

    Returns:
    -------
    noisy_text : np.ndarray of shape (N,)
        Copy of text_array with erased characters.
    erase_mask : np.ndarray of shape (N,)
        Boolean array where True indicates erased positions.
    """
    erase_mask = np.random.uniform(size=text_array.shape) < erase_rate
    noisy_text = np.copy(text_array)
    noisy_text[erase_mask] = erase_char
    return noisy_text, erase_mask
