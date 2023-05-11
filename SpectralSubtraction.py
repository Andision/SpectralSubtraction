import numpy as np
import scipy
import soundfile
import librosa
import matplotlib.pyplot as plt
import os

import xunfei as xf

MAIN_DIR = "C:/Users/Andision/Documents/GitHub/SpectralSubtraction/"
# AUDIO_TEST = MAIN_DIR+"test.wav"
# AUDIO_FILE = MAIN_DIR+"test.wav"
AUDIO_FILE = MAIN_DIR+"subway.broadcast.wav"
OUTPUT_FILE = MAIN_DIR+"output.wav"
ORIGIN_1_FILE = MAIN_DIR+"origin1.wav"
ORIGIN_2_FILE = MAIN_DIR+"origin2.wav"

NOISE_RATE_THRESHOLD = 1

alpha = 1
beta = 0.1


def Process():
    timeSeries, samplingRate = librosa.load(AUDIO_FILE, mono=False, sr=None)
    audioTimeSeries = timeSeries[1]
    noiseTimeSeries = timeSeries[0]

    audioFrequencySeries = librosa.stft(audioTimeSeries)
    audioFrequencySeriesAmplitude = np.abs(audioFrequencySeries)
    audioFrequencySeriesEnergy = np.square(audioFrequencySeriesAmplitude)
    audioFrequencySeriesPhase = np.angle(audioFrequencySeries)

    noiseFrequencySeries = librosa.stft(noiseTimeSeries)
    noiseFrequencySeriesAmplitude = np.abs(noiseFrequencySeries)
    noiseFrequencySeriesEnergy = np.square(noiseFrequencySeriesAmplitude)
    noiseFrequencySeriesPhase = np.angle(noiseFrequencySeries)

    clearFrequencySeriesEnergy = audioFrequencySeriesEnergy - \
        alpha * noiseFrequencySeriesEnergy
    mask = (clearFrequencySeriesEnergy < 0)
    clearFrequencySeriesEnergy[mask] = beta*noiseFrequencySeriesEnergy[mask]
    clearFrequencySeries = np.sqrt(clearFrequencySeriesEnergy)
    clearTimeSeries = librosa.istft(clearFrequencySeries)

    # clearFrequencySeriesAmplitude = audioFrequencySeriesAmplitude-noiseFrequencySeriesAmplitude
    # clearFrequencySeries = clearFrequencySeriesAmplitude*np.exp(1.0j* audioFrequencySeriesPhase)
    # clearTimeSeries = librosa.istft(clearFrequencySeries)

    soundfile.write(OUTPUT_FILE, clearTimeSeries, samplingRate)
    soundfile.write(ORIGIN_1_FILE, audioTimeSeries, samplingRate)
    soundfile.write(ORIGIN_2_FILE, noiseTimeSeries, samplingRate)

    fig, ax = plt.subplots(3, 1, constrained_layout=True)

    ax[0].plot(range(0, clearTimeSeries.size), clearTimeSeries, color='green')
    ax[1].plot(range(0, audioTimeSeries.size), audioTimeSeries, color='blue')
    ax[2].plot(range(0, noiseTimeSeries.size), noiseTimeSeries, color='red')

    ax[0].set_title("clear")
    ax[1].set_title("audio")
    ax[2].set_title("noise")

    # ax[0].set_ylim((-1,1))
    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Amplitude')

    plt.show()

    # for i in range(0,audioTimeSeries.size):
    #     print(i,audioTimeSeries[i],clearTimeSeries[i])


def FrequencyDomainProcess(samplePath):

    audioFileDir = samplePath+'/audio.wav'
    leftChannelTimeSeries, rightChannelTimeSeries, samplingRate = ImportAudioFile(
        audioFileDir)

    if np.sum(np.square(leftChannelTimeSeries)) > np.sum(np.square(rightChannelTimeSeries)):
        audioTimeSeries = leftChannelTimeSeries.copy()
        noiseTimeSeries = rightChannelTimeSeries.copy()
    else:
        audioTimeSeries = rightChannelTimeSeries.copy()
        noiseTimeSeries = leftChannelTimeSeries.copy()

    audioFrequencySeries = librosa.stft(audioTimeSeries)
    audioFrequencySeriesAmplitude = np.abs(audioFrequencySeries)
    audioFrequencySeriesEnergy = np.square(audioFrequencySeriesAmplitude)
    audioFrequencySeriesPhase = np.angle(audioFrequencySeries)

    noiseFrequencySeries = librosa.stft(noiseTimeSeries)
    noiseFrequencySeriesAmplitude = np.abs(noiseFrequencySeries)
    noiseFrequencySeriesEnergy = np.square(noiseFrequencySeriesAmplitude)
    noiseFrequencySeriesPhase = np.angle(noiseFrequencySeries)

    clearFrequencySeriesEnergy = audioFrequencySeriesEnergy - \
        alpha * noiseFrequencySeriesEnergy
    mask = (clearFrequencySeriesEnergy < 0)
    clearFrequencySeriesEnergy[mask] = beta*noiseFrequencySeriesEnergy[mask]
    clearFrequencySeries = np.sqrt(clearFrequencySeriesEnergy)
    clearTimeSeries = librosa.istft(clearFrequencySeries)

    soundfile.write(samplePath+'/fdp.pcm', clearTimeSeries,
                    samplingRate, subtype="PCM_16", format="RAW")
    soundfile.write(samplePath+'/fdp.wav', clearTimeSeries,
                    samplingRate, subtype="PCM_16", format="WAV")


def ImportAudioFile(fileDir: str):
    timeSeries, samplingRate = librosa.load(fileDir, mono=False, sr=None)
    # leftChannelTimeSeries  = timeSeries[1]
    # rightChannelTimeSeries = timeSeries[0]

    return timeSeries[0], timeSeries[1], samplingRate


def CalculateBalanceFactor(leftChannelTimeSeries, rightChannelTimeSeries):
    leftChannelPowerCount = 0
    rightChannelPowerCount = 0
    timeSeriesLength = leftChannelTimeSeries.shape[0]
    balanceFactorList = []

    for i in range(0, min(10000, timeSeriesLength)):
        leftChannelPowerCount += leftChannelTimeSeries[i] ** 2
        rightChannelPowerCount += rightChannelTimeSeries[i] ** 2

        if leftChannelTimeSeries[i] == 0 or rightChannelTimeSeries[i] == 0:
            continue

        balanceFactorList.append(
            leftChannelTimeSeries[i] ** 2/rightChannelTimeSeries[i] ** 2)

    balanceFactorList.sort()

    if False:
        pass

    return balanceFactorList[int(len(balanceFactorList)/2)]


def BalanceTwoChannel(leftChannelTimeSeries, rightChannelTimeSeries, balanceFactor):
    leftChannelBalancedTimeSeries = leftChannelTimeSeries.copy()
    rightChannelBalancedTimeSeries = rightChannelTimeSeries.copy()

    if balanceFactor <= 1:
        mask = (leftChannelBalancedTimeSeries < 0)
        leftChannelBalancedTimeSeries = np.sqrt(
            np.square(leftChannelBalancedTimeSeries)/balanceFactor)
        leftChannelBalancedTimeSeries[mask] *= -1

    else:
        mask = (rightChannelBalancedTimeSeries < 0)
        rightChannelBalancedTimeSeries = np.sqrt(
            np.square(rightChannelBalancedTimeSeries)*balanceFactor)
        rightChannelBalancedTimeSeries[mask] *= -1

    return leftChannelBalancedTimeSeries, rightChannelBalancedTimeSeries


def NoiseCancellation(leftChannelBalancedTimeSeries, rightChannelBalancedTimeSeries):
    if np.sum(np.square(leftChannelBalancedTimeSeries)) > np.sum(np.square(rightChannelBalancedTimeSeries)):
        masterBalancedTimeSeries = leftChannelBalancedTimeSeries.copy()
        slaveBalancedTimeSeries = rightChannelBalancedTimeSeries.copy()
    else:
        masterBalancedTimeSeries = rightChannelBalancedTimeSeries.copy()
        slaveBalancedTimeSeries = leftChannelBalancedTimeSeries.copy()

    for i in range(len(masterBalancedTimeSeries)):
        masterPower = masterBalancedTimeSeries[i]**2
        slavePower = slaveBalancedTimeSeries[i]**2

        if masterPower == 0 or slavePower == 0:
            continue

        # if slavePower > masterPower*NOISE_RATE_THRESHOLD:
        #     signHere = masterBalancedTimeSeries[i]/np.abs(masterBalancedTimeSeries[i])
        #     masterBalancedTimeSeries[i] = signHere * np.sqrt(max(masterPower-slavePower*NOISE_RATE_THRESHOLD,0))

        signHere = masterBalancedTimeSeries[i] / \
            np.abs(masterBalancedTimeSeries[i])
        masterBalancedTimeSeries[i] = signHere * \
            np.sqrt(max(masterPower-slavePower, 0))

    return masterBalancedTimeSeries


def TimeDomainProcess(samplePath):
    # audioFileDir = MAIN_DIR+"subway.broadcast.wav"
    audioFileDir = samplePath+'/audio.wav'
    leftChannelTimeSeries, rightChannelTimeSeries, samplingRate = ImportAudioFile(
        audioFileDir)

    balanceFactor = CalculateBalanceFactor(
        leftChannelTimeSeries, rightChannelTimeSeries)

    leftChannelBalancedTimeSeries, rightChannelBalancedTimeSeries = BalanceTwoChannel(
        leftChannelTimeSeries, rightChannelTimeSeries, balanceFactor)

    clearTimeSeries = NoiseCancellation(leftChannelBalancedTimeSeries,
                                        rightChannelBalancedTimeSeries)

    # soundfile.write(OUTPUT_FILE, clearTimeSeries, samplingRate)
    # soundfile.write(ORIGIN_1_FILE, leftChannelBalancedTimeSeries, samplingRate)
    # soundfile.write(ORIGIN_2_FILE, rightChannelBalancedTimeSeries, samplingRate)
    soundfile.write(samplePath+'/tdp.pcm', clearTimeSeries,
                    samplingRate, subtype="PCM_16", format="RAW")
    soundfile.write(samplePath+'/tdp.wav', clearTimeSeries,
                    samplingRate, subtype="PCM_16", format="WAV")


if __name__ == "__main__":

    rootPath = "C:/Users/Andision/Documents/GitHub/SpectralSubtraction/audioFiles/"

    for sample in os.listdir(rootPath):

        samplePath = rootPath+sample
        pcmFilePath = samplePath+'/audio.pcm'
        wavFilePath = samplePath+'/audio.wav'

        wavFileSize = os.path.getsize(wavFilePath)

        # 200KB
        if wavFileSize < 200 * 1024:
            continue

        TimeDomainProcess(samplePath)
        FrequencyDomainProcess(samplePath)
        tdpResult = xf.voiceToText(samplePath+'/tdp.pcm').strip()
        fdpResult = xf.voiceToText(samplePath+'/fdp.pcm').strip()

        # print(tdpResult)
        # print(fdpResult)

        resFilePath = samplePath+'/rec_result.txt'
        with open(resFilePath, encoding='utf-8') as f:
            resFile = f.readlines()
            baselineResult = resFile[-1].strip() if len(resFile) > 0 else ""

        if baselineResult != tdpResult or baselineResult != fdpResult:
            print(sample)
            print(baselineResult)
            print(tdpResult)
            print(fdpResult)
            print('\n')
