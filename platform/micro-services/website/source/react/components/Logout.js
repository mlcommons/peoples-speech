import React from 'react';
import { GoogleLogout } from 'react-google-login';

const clientId =
  '32760422085-n4j0v90sgp174vp3124ers94aj5qp1ms.apps.googleusercontent.com';

function Logout() {
  const onSuccess = () => {
    console.log('Logout made successfully');
    alert('Logout made successfully ✌');
  };

  return (
    <div>
      <GoogleLogout
        clientId={clientId}
        buttonText="Logout"
        onLogoutSuccess={onSuccess}
      ></GoogleLogout>
    </div>
  );
}

export default Logout;
