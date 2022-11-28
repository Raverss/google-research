import random
import numpy as np
import copy
import cv2

from dataclasses import dataclass


def segment(img, ratio, numberOfRectangles, refinementSteps):
    """Partition image into uneven blocks.

    Args:
        img (np.ndarray): Input image.
        refIter (int): Number of refiment iterations.
        ratio (float): What ratio the rectangle should have.
        refinementSteps (int): how many invalid attempts can happen before forced end.

    Returns:
        list: list of blocks [leftup.y, leftup.x, rightbottom.y, rightbottom.x]
    """

    # TODO: change the structure to the list
    @dataclass
    class Rect:
        x: float
        y: float
        width: float
        height: float

    imageWidth, imageHeight = img.shape[:2]
    # minimal side of the rectangle
    minSide = round(min(imageWidth, imageHeight) / 6)
    tolerance = ratio * 0.20  # tolerance for the ratio above 20%
    rects = []

    def split(horizontal, index, rects, minSide):
        if rects[index].height < 2 * minSide + 1 and rects[index].width < 2 * minSide + 1:
            return False

        if (rects[index].height < 2 * minSide + 1 and horizontal) or (rects[index].width < 2 * minSide + 1 and not horizontal):
            horizontal = not horizontal

        newRect = copy.copy(rects[index])

        if horizontal:
            newHeight = random.randint(minSide, rects[index].height - minSide)

            newRect.y = rects[index].y + newHeight
            newRect.height = rects[index].height - newHeight

            if (newRect.height < minSide):
                return False

            rects[index].height = newHeight
        else:
            newWidth = random.randint(minSide, rects[index].width - minSide)

            newRect.x = rects[index].x + newWidth
            newRect.width = rects[index].width - newWidth

            if (newRect.width < minSide):
                return False

            rects[index].width = newWidth

        # Add new rect to the list
        rects.append(newRect)

        return True

    # Refinement the rectangles to look more like the demanded ratio considering tolerance and prevent stretches
    def refinement(rects, ratio, tolerance, minSide):
        atLeastOneInvalid = False
        index = 0

        # We must work with the copy because "Split" alters the original list
        rectsCopy = rects.copy()

        for rect in rectsCopy:
            # Check validity
            currentRatio = max(rect.width, rect.height) / min(rect.width, rect.height)

            if currentRatio > ratio + tolerance or currentRatio < ratio - tolerance:
                atLeastOneInvalid = True

                if rect.height > rect.width:  # tall
                    split(True, index, rects, minSide)
                else:  # thick
                    split(False, index, rects, minSide)

            index = index + 1

        return atLeastOneInvalid

    rects.append(Rect(0, 0, imageWidth, imageHeight))
    # Split
    currentNumberOfRectangles = 1

    while (currentNumberOfRectangles < numberOfRectangles):
        if split(bool(random.getrandbits(1)), random.randint(0, len(rects) - 1), rects, minSide):
            currentNumberOfRectangles = currentNumberOfRectangles + 1

    currentRefinement = 0

    while currentRefinement < refinementSteps:
        if not refinement(rects, ratio, tolerance, minSide):
            break
        else:
            currentRefinement = currentRefinement + 1

    rects = [[int(rect.x), int(rect.y), int(rect.x+rect.width), int(rect.y+rect.height)] for rect in rects]
    return rects


def unproportional(data: np.ndarray, ratio: float=1.18, numberOfRectangles: int=5, refinementSteps: int=7, p: float=0.1):
    """Partition image into uneven blocks, shuffle them and re-assemble the image again.

    Args:
        img ([np.ndarray, list, tuple]): Input image, or list/tuple of images (expecting image and semantic label).
        refIter (int): Number of refinement repetetions.
        ratio (float): Block ratio during division.
        numberOfRectangles (int): Number of starting rectangles.
        refinementSteps (int): Number of refinement steps in one iteration. 

    Returns:
        np.ndarray: Unproportionally shuffled image.
    """
    
    if random.random() > p:
        return data

    assert data.ndim <= 3, "batched data"

    blocks = segment(data, ratio, numberOfRectangles, refinementSteps)

    ind = np.linspace(0, len(blocks)-1, len(blocks), dtype=np.int32).tolist()
    random.shuffle(ind)

    def swap(img):
        """Swap blocks in the image around, according shuffled"""
        img = img.copy()
        while len(ind) > 1:
            b1_id = ind.pop(0)
            b2_id = ind.pop(0)

            b1 = img[blocks[b1_id][1]:blocks[b1_id][3], blocks[b1_id][0]:blocks[b1_id][2], ...]
            b1 = cv2.resize(b1, (blocks[b2_id][2]-blocks[b2_id][0], blocks[b2_id][3]-blocks[b2_id][1]), interpolation=cv2.INTER_CUBIC)

            b2 = img[blocks[b2_id][1]:blocks[b2_id][3], blocks[b2_id][0]:blocks[b2_id][2], ...]
            b2 = cv2.resize(b2, (blocks[b1_id][2]-blocks[b1_id][0], blocks[b1_id][3]-blocks[b1_id][1]), interpolation=cv2.INTER_CUBIC)

            img[blocks[b1_id][1]:blocks[b1_id][3], blocks[b1_id][0]:blocks[b1_id][2], ...] = b2
            img[blocks[b2_id][1]:blocks[b2_id][3], blocks[b2_id][0]:blocks[b2_id][2], ...] = b1

        return img
    
    if isinstance(data, np.ndarray):
        data = swap(data)
        return data
    else:
        raise TypeError(f"data must be numpy array but is {type(data)}")


class Unproportional(object):
    def __init__(self, ratio=4.0/3.0, numberOfRectangles=2, refinementSteps=10, p=1):
        self.ratio = ratio
        self.num_rect = numberOfRectangles
        self.ref_step = refinementSteps
        self.p = p

    def __call__(self, data):
        return unproportional(data, self.ratio, self.num_rect, self.ref_step, self.p)