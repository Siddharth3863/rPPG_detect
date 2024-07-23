import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from './components/Navbar.js';
import RecordSessionPFE from './RecordSessionPFE';
import RecordSessionPhys from './RecordSessionPhys';
import RecordSessionSSL from './RecordSessionSSL';
import 'bootstrap/dist/css/bootstrap.min.css';
import HomePage from "./components/Home";


const App = () => {
  return (
      <Router>
          <Switch>
              <Route exact path="/" component={HomePage} />
              <Route path="/record-session-pfe" component={RecordSessionPFE} />
              <Route path="/record-session-phys" component={RecordSessionPhys} />
              <Route path="/record-session-ssl" component={RecordSessionSSL} />
          </Switch>
      </Router>
  );
};

export default App;


