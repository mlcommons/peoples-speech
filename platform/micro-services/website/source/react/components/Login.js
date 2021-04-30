import React, { Component } from 'react';

import { GoogleLogin } from 'react-google-login';
// refresh token
import { refreshTokenSetup } from '../utils/refreshToken';

const clientId =
  '32760422085-n4j0v90sgp174vp3124ers94aj5qp1ms.apps.googleusercontent.com';

class Login extends Component {

    render() {
        const onSuccess = (res) => {
            console.log('Login Success: currentUser:', res.profileObj);
            alert(
                `Logged in successfully welcome ${res.profileObj.name} 😍. \n See console for full profile object.`
            );
            refreshTokenSetup(res);
            fetch('http://localhost:5000/peoples_speech/train_url',
                {
                    method: 'POST', // *GET, POST, PUT, DELETE, etc.
                    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
                    headers: {
                      'Content-Type': 'application/json'
                      // 'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: JSON.stringify(res.profileObj) // body data type must match "Content-Type" header
                }
            )
            .then(res => res.json())
            .then((data) => {
                console.log("Got response: ", data);
                this.props.addTrainURL(data["url"])
            })
            .catch(console.log)
            fetch('http://localhost:5000/peoples_speech/dev_url',
                {
                    method: 'POST', // *GET, POST, PUT, DELETE, etc.
                    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
                    headers: {
                      'Content-Type': 'application/json'
                      // 'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: JSON.stringify(res.profileObj) // body data type must match "Content-Type" header
                }
            )
            .then(res => res.json())
            .then((data) => {
                console.log("Got response: ", data);
                this.props.addDevURL(data["url"])
            })
            .catch(console.log)
            fetch('http://localhost:5000/peoples_speech/test_url',
                {
                    method: 'POST', // *GET, POST, PUT, DELETE, etc.
                    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
                    headers: {
                      'Content-Type': 'application/json'
                      // 'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: JSON.stringify(res.profileObj) // body data type must match "Content-Type" header
                }
            )
            .then(res => res.json())
            .then((data) => {
                console.log("Got response: ", data);
                this.props.addTestURL(data["url"])
            })
            .catch(console.log)
        };

        const onFailure = (res) => {
            console.log('Login failed: res:', res);
            alert(
                `Failed to login. 😢 Please ping this to repo owner`
            );
        };

        return (
            <div>
              <GoogleLogin
                clientId={clientId}
                buttonText="Login"
                onSuccess={onSuccess}
                onFailure={onFailure}
                cookiePolicy={'single_host_origin'}
                style={{ marginTop: '100px' }}
                isSignedIn={true}
              />
            </div>
            );
    }
}

export default Login;
