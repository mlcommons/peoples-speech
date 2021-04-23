
import config from "./config/default.json"

// Get document, or throw exception on error
console.log(config);

export default function getTrainingSetPath() {
    return config["peoples_speech"]["train_path"]
}
