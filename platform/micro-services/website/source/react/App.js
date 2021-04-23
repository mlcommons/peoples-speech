import logo from './logo.png';
import './App.css';
//import getGoogleCloudStorageUrl from './getSignUrl.js';
//import getTrainingSetPath from './getTrainingSetPath.js';

import React, { Component } from 'react';

class App extends Component {

    state = {
        train_dataset_url: [],
        dev_dataset_url: [],
        test_dataset_url: []
      }

    render() {
      return (
        <div className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
            <p>Download the People's Speech Dataset</p>
            <a
              className="App-link"
              href={this.state.train_dataset_url}
              target="_blank"
              rel="noopener noreferrer"
            >
            Train
            </a>
            <a
              className="App-link"
              href={this.state.dev_dataset_url}
              target="_blank"
              rel="noopener noreferrer"
            >
            Development
            </a>
            <a
              className="App-link"
              href={this.state.test_dataset_url}
              target="_blank"
              rel="noopener noreferrer"
            >
            Test
            </a>
          </header>
        </div>
      );
    }

    componentDidMount() {
        fetch('http://localhost:5000/peoples_speech/train_url')
        .then(res => res.json())
        .then((data) => {
          console.log("Got response: ", data);
          this.setState({ train_dataset_url : data["url"] })
        })
        .catch(console.log)
        fetch('http://localhost:5000/peoples_speech/dev_url')
        .then(res => res.json())
        .then((data) => {
          console.log("Got response: ", data);
          this.setState({ dev_dataset_url : data["url"] })
        })
        .catch(console.log)
        fetch('http://localhost:5000/peoples_speech/test_url')
        .then(res => res.json())
        .then((data) => {
          console.log("Got response: ", data);
          this.setState({ test_dataset_url : data["url"] })
        })
        .catch(console.log)
      }
}

export default App;
