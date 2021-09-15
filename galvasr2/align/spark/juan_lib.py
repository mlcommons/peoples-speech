WAV_FROM_MP3_TYPE = T.StructType(
    [
        T.StructField("audio_name", T.StringType()),
        T.StructField("audio", T.BinaryType()),
    ]
)


@F.pandas_udf(WAV_FROM_MP3_TYPE)
def wav_from_mp3_udf(
    audio_bytes_series: pd.Series,
    audio_type_series: pd.Series,
    audio_name_series: pd.Series,
) -> pd.DataFrame:
    output_array = []
    for audio_bytes, audio_type, audio_name in zip(
        audio_bytes_series, audio_type_series, audio_name_series
    ):
        assert audio_type == "mp3"
        decoded_bytes = DecodeToWavPipe(audio_bytes, audio_type)
        new_name = re.substitute(audio_name, "[.]mp3$", ".wav")
        output_array.append({"audio_name": new_name, "audio": decoded_bytes})
    return pd.DataFrame(output_array)


def create_my_udf(common_params):
    @F.pandas_udf(RETURN_TYPE)
    def my_udf():
        common_params
        pass

    return my_udf
