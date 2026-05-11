import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()

def listen_and_control():

    # 🔥 Faster engine init
    engine = pyttsx3.init()
    
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)

    engine.setProperty('rate', 185)   # 🔥 faster speech

    action_phrase = "No command recognized"
    command = ""

    try:
        with sr.Microphone() as source:
            print("Listening...")
            
            # 🔥 REDUCED calibration time
            r.adjust_for_ambient_noise(source, duration=0.2)
            
            audio = r.listen(source, timeout=4, phrase_time_limit=3)

        command = r.recognize_google(audio).lower()
        print("Recognized:", command)

    except sr.UnknownValueError:
        return "Not recognized"

    except sr.RequestError:
        return "API error"

    except sr.WaitTimeoutError:
        return "No speech detected"

    # 🔹 Command mapping
    if "move forward" in command:
        action_phrase = "Moving forward"

    elif "move backward" in command:
        action_phrase = "Moving backward"

    elif "turn left" in command:
        action_phrase = "Turning left"

    elif "turn right" in command:
        action_phrase = "Turning right"

    elif "stop the car" in command:
        action_phrase = "Stopping"

    elif "self parking" in command:
        action_phrase = "Self parking mode activated"

    elif "intruder" in command:
        action_phrase = "Intruder mode activated"

    elif "shutdown" in command or "exit" in command:
        action_phrase = "Shutting down"

    else:
        action_phrase = "Command not recognized"

    # 🔥 Faster speaking (no queue buildup)
    engine.say(action_phrase)
    engine.runAndWait()
    engine.stop()

    return action_phrase