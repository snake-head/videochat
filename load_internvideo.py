from intern_action import intern_action_b16
from huggingface_hub import hf_hub_download
# from kinetics_class_index import kinetics_classnames
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

class Intern_Action(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
    
    def forward(self, x):
        return self.backbone(x)

def get_index(num_frames, num_segments=8):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def transform_action():
    # transform
    crop_size = 224
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    return T.Compose([
        # T.ToPILImage(),
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

def load_intern_action(device):
    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[k] = v

    # model_path = hf_hub_download(repo_id="Andy1621/uniformerv2", filename="k400+k710_uniformerv2_b16_8x224.pyth")
    model_path = "/mnt/data.coronaryct.1/ZhuYichen/Ask-Anything/video_chat_with_ChatGPT/pretrained_models/uniformerv2//k400+k710_uniformerv2_b16_8x224.pyth"
    # Pick a pretrained model 
    model = Intern_Action(intern_action_b16(pretrained=False, t_size=8, no_lmhra=True, temporal_downsample=False))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()
    return model

def cut_frame_to_8(data):
    index = np.linspace(0, len(data)-1, 8).astype(int)
    return data[index]

kinetics_classnames = {
    "0": "riding a bike", 
    "1": "marching", 
    "2": "dodgeball", 
    "3": "playing cymbals", 
    "4": "checking tires", 
    "5": "roller skating", 
    "6": "tasting beer", 
    "7": "clapping", 
    "8": "drawing", 
    "9": "juggling fire", 
    "10": "bobsledding", 
    "11": "petting animal (not cat)", 
    "12": "spray painting", 
    "13": "training dog", 
    "14": "eating watermelon", 
    "15": "building cabinet", 
    "16": "applauding", 
    "17": "playing harp", 
    "18": "balloon blowing", 
    "19": "sled dog racing", 
    "20": "wrestling", 
    "21": "pole vault", 
    "22": "hurling (sport)", 
    "23": "riding scooter", 
    "24": "shearing sheep", 
    "25": "sweeping floor", 
    "26": "eating carrots", 
    "27": "skateboarding", 
    "28": "dunking basketball", 
    "29": "disc golfing", 
    "30": "eating spaghetti", 
    "31": "playing flute", 
    "32": "riding mechanical bull", 
    "33": "making sushi", 
    "34": "trapezing", 
    "35": "picking fruit", 
    "36": "stretching leg", 
    "37": "playing ukulele", 
    "38": "tying tie", 
    "39": "skydiving", 
    "40": "playing cello", 
    "41": "jumping into pool", 
    "42": "shooting goal (soccer)", 
    "43": "trimming trees", 
    "44": "bookbinding", 
    "45": "ski jumping", 
    "46": "walking the dog", 
    "47": "riding unicycle", 
    "48": "shaving head", 
    "49": "hopscotch", 
    "50": "playing piano", 
    "51": "parasailing", 
    "52": "bartending", 
    "53": "kicking field goal", 
    "54": "finger snapping", 
    "55": "dining", 
    "56": "yawning", 
    "57": "peeling potatoes", 
    "58": "canoeing or kayaking", 
    "59": "front raises", 
    "60": "laughing", 
    "61": "dancing macarena", 
    "62": "digging", 
    "63": "reading newspaper", 
    "64": "hitting baseball", 
    "65": "clay pottery making", 
    "66": "exercising with an exercise ball", 
    "67": "playing saxophone", 
    "68": "shooting basketball", 
    "69": "washing hair", 
    "70": "lunge", 
    "71": "brushing hair", 
    "72": "curling hair", 
    "73": "kitesurfing", 
    "74": "tapping guitar", 
    "75": "bending back", 
    "76": "skipping rope", 
    "77": "situp", 
    "78": "folding paper", 
    "79": "cracking neck", 
    "80": "assembling computer", 
    "81": "cleaning gutters", 
    "82": "blowing out candles", 
    "83": "shaking hands", 
    "84": "dancing gangnam style", 
    "85": "windsurfing", 
    "86": "tap dancing", 
    "87": "skiing (not slalom or crosscountry)", 
    "88": "bandaging", 
    "89": "push up", 
    "90": "doing nails", 
    "91": "punching person (boxing)", 
    "92": "bouncing on trampoline", 
    "93": "scrambling eggs", 
    "94": "singing", 
    "95": "cleaning floor", 
    "96": "krumping", 
    "97": "drumming fingers", 
    "98": "snowmobiling", 
    "99": "gymnastics tumbling", 
    "100": "headbanging", 
    "101": "catching or throwing frisbee", 
    "102": "riding elephant", 
    "103": "bee keeping", 
    "104": "feeding birds", 
    "105": "snatch weight lifting", 
    "106": "mowing lawn", 
    "107": "fixing hair", 
    "108": "playing trumpet", 
    "109": "flying kite", 
    "110": "crossing river", 
    "111": "swinging legs", 
    "112": "sanding floor", 
    "113": "belly dancing", 
    "114": "sneezing", 
    "115": "clean and jerk", 
    "116": "side kick", 
    "117": "filling eyebrows", 
    "118": "shuffling cards", 
    "119": "recording music", 
    "120": "cartwheeling", 
    "121": "feeding fish", 
    "122": "folding clothes", 
    "123": "water skiing", 
    "124": "tobogganing", 
    "125": "blowing leaves", 
    "126": "smoking", 
    "127": "unboxing", 
    "128": "tai chi", 
    "129": "waxing legs", 
    "130": "riding camel", 
    "131": "slapping", 
    "132": "tossing salad", 
    "133": "capoeira", 
    "134": "playing cards", 
    "135": "playing organ", 
    "136": "playing violin", 
    "137": "playing drums", 
    "138": "tapping pen", 
    "139": "vault", 
    "140": "shoveling snow", 
    "141": "playing tennis", 
    "142": "getting a tattoo", 
    "143": "making a sandwich", 
    "144": "making tea", 
    "145": "grinding meat", 
    "146": "squat", 
    "147": "eating doughnuts", 
    "148": "ice fishing", 
    "149": "snowkiting", 
    "150": "kicking soccer ball", 
    "151": "playing controller", 
    "152": "giving or receiving award", 
    "153": "welding", 
    "154": "throwing discus", 
    "155": "throwing axe", 
    "156": "ripping paper", 
    "157": "swimming butterfly stroke", 
    "158": "air drumming", 
    "159": "blowing nose", 
    "160": "hockey stop", 
    "161": "taking a shower", 
    "162": "bench pressing", 
    "163": "planting trees", 
    "164": "pumping fist", 
    "165": "climbing tree", 
    "166": "tickling", 
    "167": "high kick", 
    "168": "waiting in line", 
    "169": "slacklining", 
    "170": "tango dancing", 
    "171": "hurdling", 
    "172": "carrying baby", 
    "173": "celebrating", 
    "174": "sharpening knives", 
    "175": "passing American football (in game)", 
    "176": "headbutting", 
    "177": "playing recorder", 
    "178": "brush painting", 
    "179": "garbage collecting", 
    "180": "robot dancing", 
    "181": "shredding paper", 
    "182": "pumping gas", 
    "183": "rock climbing", 
    "184": "hula hooping", 
    "185": "braiding hair", 
    "186": "opening present", 
    "187": "texting", 
    "188": "decorating the christmas tree", 
    "189": "answering questions", 
    "190": "playing keyboard", 
    "191": "writing", 
    "192": "bungee jumping", 
    "193": "sniffing", 
    "194": "eating burger", 
    "195": "playing accordion", 
    "196": "making pizza", 
    "197": "playing volleyball", 
    "198": "tasting food", 
    "199": "pushing cart", 
    "200": "spinning poi", 
    "201": "cleaning windows", 
    "202": "arm wrestling", 
    "203": "changing oil", 
    "204": "swimming breast stroke", 
    "205": "tossing coin", 
    "206": "deadlifting", 
    "207": "hoverboarding", 
    "208": "cutting watermelon", 
    "209": "cheerleading", 
    "210": "snorkeling", 
    "211": "washing hands", 
    "212": "eating cake", 
    "213": "pull ups", 
    "214": "surfing water", 
    "215": "eating hotdog", 
    "216": "holding snake", 
    "217": "playing harmonica", 
    "218": "ironing", 
    "219": "cutting nails", 
    "220": "golf chipping", 
    "221": "shot put", 
    "222": "hugging", 
    "223": "playing clarinet", 
    "224": "faceplanting", 
    "225": "trimming or shaving beard", 
    "226": "drinking shots", 
    "227": "riding mountain bike", 
    "228": "tying bow tie", 
    "229": "swinging on something", 
    "230": "skiing crosscountry", 
    "231": "unloading truck", 
    "232": "cleaning pool", 
    "233": "jogging", 
    "234": "ice climbing", 
    "235": "mopping floor", 
    "236": "making bed", 
    "237": "diving cliff", 
    "238": "washing dishes", 
    "239": "grooming dog", 
    "240": "weaving basket", 
    "241": "frying vegetables", 
    "242": "stomping grapes", 
    "243": "moving furniture", 
    "244": "cooking sausages", 
    "245": "doing laundry", 
    "246": "dying hair", 
    "247": "knitting", 
    "248": "reading book", 
    "249": "baby waking up", 
    "250": "punching bag", 
    "251": "surfing crowd", 
    "252": "cooking chicken", 
    "253": "pushing car", 
    "254": "springboard diving", 
    "255": "swing dancing", 
    "256": "massaging legs", 
    "257": "beatboxing", 
    "258": "breading or breadcrumbing", 
    "259": "somersaulting", 
    "260": "brushing teeth", 
    "261": "stretching arm", 
    "262": "juggling balls", 
    "263": "massaging person's head", 
    "264": "eating ice cream", 
    "265": "extinguishing fire", 
    "266": "hammer throw", 
    "267": "whistling", 
    "268": "crawling baby", 
    "269": "using remote controller (not gaming)", 
    "270": "playing cricket", 
    "271": "opening bottle", 
    "272": "playing xylophone", 
    "273": "motorcycling", 
    "274": "driving car", 
    "275": "exercising arm", 
    "276": "passing American football (not in game)", 
    "277": "playing kickball", 
    "278": "sticking tongue out", 
    "279": "flipping pancake", 
    "280": "catching fish", 
    "281": "eating chips", 
    "282": "shaking head", 
    "283": "sword fighting", 
    "284": "playing poker", 
    "285": "cooking on campfire", 
    "286": "doing aerobics", 
    "287": "paragliding", 
    "288": "using segway", 
    "289": "folding napkins", 
    "290": "playing bagpipes", 
    "291": "gargling", 
    "292": "skiing slalom", 
    "293": "strumming guitar", 
    "294": "javelin throw", 
    "295": "waxing back", 
    "296": "riding or walking with horse", 
    "297": "plastering", 
    "298": "long jump", 
    "299": "parkour", 
    "300": "wrapping present", 
    "301": "egg hunting", 
    "302": "archery", 
    "303": "cleaning toilet", 
    "304": "swimming backstroke", 
    "305": "snowboarding", 
    "306": "catching or throwing baseball", 
    "307": "massaging back", 
    "308": "blowing glass", 
    "309": "playing guitar", 
    "310": "playing chess", 
    "311": "golf driving", 
    "312": "presenting weather forecast", 
    "313": "rock scissors paper", 
    "314": "high jump", 
    "315": "baking cookies", 
    "316": "using computer", 
    "317": "washing feet", 
    "318": "arranging flowers", 
    "319": "playing bass guitar", 
    "320": "spraying", 
    "321": "cutting pineapple", 
    "322": "waxing chest", 
    "323": "auctioning", 
    "324": "jetskiing", 
    "325": "drinking", 
    "326": "busking", 
    "327": "playing monopoly", 
    "328": "salsa dancing", 
    "329": "waxing eyebrows", 
    "330": "watering plants", 
    "331": "zumba", 
    "332": "chopping wood", 
    "333": "pushing wheelchair", 
    "334": "carving pumpkin", 
    "335": "building shed", 
    "336": "making jewelry", 
    "337": "catching or throwing softball", 
    "338": "bending metal", 
    "339": "ice skating", 
    "340": "dancing charleston", 
    "341": "abseiling", 
    "342": "climbing a rope", 
    "343": "crying", 
    "344": "cleaning shoes", 
    "345": "dancing ballet", 
    "346": "driving tractor", 
    "347": "triple jump", 
    "348": "throwing ball", 
    "349": "getting a haircut", 
    "350": "running on treadmill", 
    "351": "climbing ladder", 
    "352": "blasting sand", 
    "353": "playing trombone", 
    "354": "drop kicking", 
    "355": "country line dancing", 
    "356": "changing wheel", 
    "357": "feeding goats", 
    "358": "tying knot (not on a tie)", 
    "359": "setting table", 
    "360": "shaving legs", 
    "361": "kissing", 
    "362": "riding mule", 
    "363": "counting money", 
    "364": "laying bricks", 
    "365": "barbequing", 
    "366": "news anchoring", 
    "367": "smoking hookah", 
    "368": "cooking egg", 
    "369": "peeling apples", 
    "370": "yoga", 
    "371": "sharpening pencil", 
    "372": "dribbling basketball", 
    "373": "petting cat", 
    "374": "playing ice hockey", 
    "375": "milking cow", 
    "376": "shining shoes", 
    "377": "juggling soccer ball", 
    "378": "scuba diving", 
    "379": "playing squash or racquetball", 
    "380": "drinking beer", 
    "381": "sign language interpreting", 
    "382": "playing basketball", 
    "383": "breakdancing", 
    "384": "testifying", 
    "385": "making snowman", 
    "386": "golf putting", 
    "387": "playing didgeridoo", 
    "388": "biking through snow", 
    "389": "sailing", 
    "390": "jumpstyle dancing", 
    "391": "water sliding", 
    "392": "grooming horse", 
    "393": "massaging feet", 
    "394": "playing paintball", 
    "395": "making a cake", 
    "396": "bowling", 
    "397": "contact juggling", 
    "398": "applying cream", 
    "399": "playing badminton"
}

