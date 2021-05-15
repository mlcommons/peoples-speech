import logo from './logo.png';
import './App.css';
import Login from './components/Login.js'
import Logout from './components/Logout.js'

import React, { Component } from 'react';

class App extends Component {

    state = {
        train_dataset_url: [],
        dev_dataset_url: [],
        test_dataset_url: [],
        logged_in: false
      }

    render() {
      return (
        <div className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
            <p>Download the People's Speech Dataset</p>
            <Login
                addTrainURL={this.addTrainURL.bind(this)}
                addDevURL={this.addDevURL.bind(this)}
                addTestURL={this.addTestURL.bind(this)}
             />
            <Logout />
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

    addTrainURL(url) {
        this.setState({ train_dataset_url : url });
    }
    addDevURL(url) {
        this.setState({ dev_dataset_url : url });
    }
    addTestURL(url) {
        this.setState({ test_dataset_url : url });
    }

}

export default App;
